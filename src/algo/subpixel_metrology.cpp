#include "wafer_aoi/subpixel_metrology.hpp"
#include "wafer_aoi/image_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>

namespace wafer_aoi {

namespace {
// Solve 3x3 linear system A * x = b using Gaussian elimination with partial pivoting
bool solve_3x3(double A[3][3], double b[3], double x[3]) {
    double M[3][4];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] = A[i][j];
        M[i][3] = b[i];
    }
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        double max_val = std::abs(M[col][col]);
        for (int row = col + 1; row < 3; ++row) {
            if (std::abs(M[row][col]) > max_val) {
                max_val = std::abs(M[row][col]);
                pivot = row;
            }
        }
        if (max_val < 1e-12) return false;
        if (pivot != col) {
            for (int j = col; j < 4; ++j) std::swap(M[col][j], M[pivot][j]);
        }
        for (int row = col + 1; row < 3; ++row) {
            double f = M[row][col] / M[col][col];
            for (int j = col; j < 4; ++j) M[row][j] -= f * M[col][j];
        }
    }
    for (int i = 2; i >= 0; --i) {
        double sum = M[i][3];
        for (int j = i + 1; j < 3; ++j) sum -= M[i][j] * x[j];
        x[i] = sum / M[i][i];
    }
    return true;
}
}  // namespace

std::vector<SubpixelPoint> extract_subpixel_edges(
    const ImageF32& img, float mag_thresh, float smooth_sigma,
    int roi_x, int roi_y, int roi_w, int roi_h) {
    if (img.empty()) return {};
    ImageF32 smooth = gaussian_blur(img, smooth_sigma);
    ImageF32 gx, gy, mag;
    sobel_gradients(smooth, gx, gy, mag);

    int x0 = std::max(2, roi_x);
    int y0 = std::max(2, roi_y);
    int x1 = (roi_w > 0) ? std::min(img.width() - 2, roi_x + roi_w) : (img.width() - 2);
    int y1 = (roi_h > 0) ? std::min(img.height() - 2, roi_y + roi_h) : (img.height() - 2);

    std::vector<SubpixelPoint> pts;
    pts.reserve(static_cast<size_t>(std::max(0, (x1 - x0) * (y1 - y0) / 16)));

    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            float m0 = mag.at(x, y);
            if (m0 < mag_thresh) continue;
            float nx = gx.at(x, y) / m0;
            float ny = gy.at(x, y) / m0;

            float m_neg = bilinear_sample(mag, x - nx, y - ny);
            float m_pos = bilinear_sample(mag, x + nx, y + ny);

            // Directional non-maximum suppression along gradient normal
            if (m0 >= m_neg && m0 >= m_pos && (m0 > m_neg || m0 > m_pos)) {
                float denom = m_neg - 2.0f * m0 + m_pos;
                float delta = 0.0f;
                if (denom < -1e-6f) {
                    delta = std::clamp(0.5f * (m_neg - m_pos) / denom, -0.5f, 0.5f);
                }
                pts.push_back(SubpixelPoint{
                    static_cast<float>(x) + delta * nx,
                    static_cast<float>(y) + delta * ny,
                    nx, ny, m0});
            }
        }
    }
    return pts;
}

Line2D fit_line_tls(const std::vector<SubpixelPoint>& pts) {
    Line2D line;
    if (pts.size() < 2) return line;

    double mx = 0.0, my = 0.0;
    for (const auto& p : pts) {
        mx += p.x;
        my += p.y;
    }
    mx /= static_cast<double>(pts.size());
    my /= static_cast<double>(pts.size());

    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (const auto& p : pts) {
        double dx = p.x - mx;
        double dy = p.y - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }

    // Normal vector = eigenvector of smallest eigenvalue of covariance matrix
    double theta = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
    // Direction of largest variance is (cos(theta), sin(theta)), so normal is (-sin, cos)
    double a = -std::sin(theta);
    double b = std::cos(theta);
    double norm = std::hypot(a, b);
    if (norm < 1e-12) return line;
    a /= norm;
    b /= norm;
    double c = -(a * mx + b * my);

    // Canonical sign convention
    if (a < 0.0 || (std::abs(a) < 1e-9 && b < 0.0)) {
        a = -a;
        b = -b;
        c = -c;
    }

    double sq_err = 0.0;
    for (const auto& p : pts) {
        double d = a * p.x + b * p.y + c;
        sq_err += d * d;
    }

    line.a = static_cast<float>(a);
    line.b = static_cast<float>(b);
    line.c = static_cast<float>(c);
    line.rms_error = static_cast<float>(std::sqrt(sq_err / static_cast<double>(pts.size())));
    line.inliers = static_cast<int>(pts.size());
    line.valid = true;
    return line;
}

Line2D fit_line_ransac(const std::vector<SubpixelPoint>& pts,
                       float inlier_thresh, int max_iters, uint32_t seed) {
    if (pts.size() < 2) return Line2D{};
    if (pts.size() == 2) return fit_line_tls(pts);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, pts.size() - 1);

    int best_inliers = -1;
    Line2D best_model;

    for (int iter = 0; iter < max_iters; ++iter) {
        size_t i1 = dist(rng);
        size_t i2 = dist(rng);
        if (i1 == i2) continue;

        double dx = pts[i2].x - pts[i1].x;
        double dy = pts[i2].y - pts[i1].y;
        double len = std::hypot(dx, dy);
        if (len < 1.0) continue;

        double a = -dy / len;
        double b = dx / len;
        double c = -(a * pts[i1].x + b * pts[i1].y);

        int count = 0;
        for (const auto& p : pts) {
            if (std::abs(a * p.x + b * p.y + c) <= inlier_thresh) {
                ++count;
            }
        }
        if (count > best_inliers) {
            best_inliers = count;
            best_model.a = static_cast<float>(a);
            best_model.b = static_cast<float>(b);
            best_model.c = static_cast<float>(c);
            best_model.valid = true;
        }
    }

    if (!best_model.valid) return Line2D{};

    std::vector<SubpixelPoint> inlier_pts;
    inlier_pts.reserve(static_cast<size_t>(best_inliers));
    for (const auto& p : pts) {
        if (best_model.distance(p.x, p.y) <= inlier_thresh) {
            inlier_pts.push_back(p);
        }
    }
    return fit_line_tls(inlier_pts);
}

Circle2D fit_circle_geometric(const std::vector<SubpixelPoint>& pts, int gn_iters) {
    Circle2D circle;
    if (pts.size() < 3) return circle;

    // Step 1: Mean-centered Kasa algebraic fit for robust initialization
    double mx = 0.0, my = 0.0;
    for (const auto& p : pts) {
        mx += p.x;
        my += p.y;
    }
    mx /= static_cast<double>(pts.size());
    my /= static_cast<double>(pts.size());

    double A[3][3] = {{0}};
    double b[3] = {0};
    for (const auto& p : pts) {
        double u = p.x - mx;
        double v = p.y - my;
        double r2 = u * u + v * v;
        A[0][0] += u * u;
        A[0][1] += u * v;
        A[0][2] += u;
        A[1][1] += v * v;
        A[1][2] += v;
        A[2][2] += 1.0;
        b[0] += -u * r2;
        b[1] += -v * r2;
        b[2] += -r2;
    }
    A[1][0] = A[0][1];
    A[2][0] = A[0][2];
    A[2][1] = A[1][2];

    double sol[3] = {0};
    if (!solve_3x3(A, b, sol)) return circle;

    double cx = mx - 0.5 * sol[0];
    double cy = my - 0.5 * sol[1];
    double r2 = 0.25 * (sol[0] * sol[0] + sol[1] * sol[1]) - sol[2];
    if (r2 <= 1e-6) return circle;
    double r = std::sqrt(r2);

    // Step 2: Gauss-Newton geometric refinement minimizing sum (dist(p, c) - r)^2
    for (int it = 0; it < gn_iters; ++it) {
        double JTJ[3][3] = {{0}};
        double JTr[3] = {0};
        for (const auto& p : pts) {
            double dx = p.x - cx;
            double dy = p.y - cy;
            double d = std::hypot(dx, dy);
            if (d < 1e-9) continue;
            double res = d - r;
            double j0 = -dx / d;
            double j1 = -dy / d;
            double j2 = -1.0;
            double J[3] = {j0, j1, j2};
            for (int r_i = 0; r_i < 3; ++r_i) {
                JTr[r_i] += J[r_i] * res;
                for (int c_i = 0; c_i < 3; ++c_i) {
                    JTJ[r_i][c_i] += J[r_i] * J[c_i];
                }
            }
        }
        // Mild Levenberg-Marquardt damping for numerical stability
        for (int k = 0; k < 3; ++k) JTJ[k][k] += 1e-6;
        double neg_JTr[3] = {-JTr[0], -JTr[1], -JTr[2]};
        double delta[3] = {0};
        if (!solve_3x3(JTJ, neg_JTr, delta)) break;

        cx += delta[0];
        cy += delta[1];
        r += delta[2];
        if (std::hypot(delta[0], delta[1]) < 1e-7 && std::abs(delta[2]) < 1e-7) {
            break;
        }
    }

    if (r <= 0.0) return circle;
    double sq_err = 0.0;
    for (const auto& p : pts) {
        double e = std::hypot(p.x - cx, p.y - cy) - r;
        sq_err += e * e;
    }

    circle.cx = static_cast<float>(cx);
    circle.cy = static_cast<float>(cy);
    circle.radius = static_cast<float>(r);
    circle.rms_error = static_cast<float>(std::sqrt(sq_err / static_cast<double>(pts.size())));
    circle.inliers = static_cast<int>(pts.size());
    circle.valid = true;
    return circle;
}

Circle2D fit_circle_ransac(const std::vector<SubpixelPoint>& pts,
                           float inlier_thresh, int max_iters, uint32_t seed) {
    if (pts.size() < 3) return Circle2D{};

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, pts.size() - 1);

    int best_inliers = -1;
    double best_cx = 0.0, best_cy = 0.0, best_r = 0.0;

    for (int iter = 0; iter < max_iters; ++iter) {
        size_t i1 = dist(rng), i2 = dist(rng), i3 = dist(rng);
        if (i1 == i2 || i2 == i3 || i1 == i3) continue;

        double x1 = pts[i1].x, y1 = pts[i1].y;
        double x2 = pts[i2].x, y2 = pts[i2].y;
        double x3 = pts[i3].x, y3 = pts[i3].y;

        double d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
        if (std::abs(d) < 1e-4) continue;

        double x1s = x1 * x1 + y1 * y1;
        double x2s = x2 * x2 + y2 * y2;
        double x3s = x3 * x3 + y3 * y3;

        double cx = (x1s * (y2 - y3) + x2s * (y3 - y1) + x3s * (y1 - y2)) / d;
        double cy = (x1s * (x3 - x2) + x2s * (x1 - x3) + x3s * (x2 - x1)) / d;
        double r = std::hypot(x1 - cx, y1 - cy);
        if (r < 2.0 || r > 5000.0) continue;

        int count = 0;
        for (const auto& p : pts) {
            if (std::abs(std::hypot(p.x - cx, p.y - cy) - r) <= inlier_thresh) {
                ++count;
            }
        }
        if (count > best_inliers) {
            best_inliers = count;
            best_cx = cx;
            best_cy = cy;
            best_r = r;
        }
    }

    if (best_inliers < 3) return Circle2D{};

    std::vector<SubpixelPoint> inliers;
    inliers.reserve(static_cast<size_t>(best_inliers));
    for (const auto& p : pts) {
        if (std::abs(std::hypot(p.x - best_cx, p.y - best_cy) - best_r) <= inlier_thresh) {
            inliers.push_back(p);
        }
    }
    return fit_circle_geometric(inliers);
}

float measure_parallel_line_pitch(const Line2D& l1, const Line2D& l2,
                                  float ref_x, float ref_y) {
    if (!l1.valid || !l2.valid) return -1.0f;
    // Project reference point onto l1, then measure perpendicular distance to l2
    float d1 = l1.signed_distance(ref_x, ref_y);
    float px = ref_x - d1 * l1.a;
    float py = ref_y - d1 * l1.b;
    return l2.distance(px, py);
}

float measure_circle_center_dist(const Circle2D& c1, const Circle2D& c2) {
    if (!c1.valid || !c2.valid) return -1.0f;
    return std::hypot(c1.cx - c2.cx, c1.cy - c2.cy);
}

RepeatabilityStats compute_repeatability(const std::vector<float>& measurements,
                                         float ground_truth) {
    RepeatabilityStats stats;
    if (measurements.empty()) return stats;
    double sum = 0.0;
    float max_err = 0.0f;
    for (float v : measurements) {
        sum += v;
        max_err = std::max(max_err, std::abs(v - ground_truth));
    }
    double mean = sum / static_cast<double>(measurements.size());
    double var = 0.0;
    for (float v : measurements) {
        double d = v - mean;
        var += d * d;
    }
    double denom = (measurements.size() > 1) ? static_cast<double>(measurements.size() - 1) : 1.0;
    double sd = std::sqrt(var / denom);

    stats.mean = static_cast<float>(mean);
    stats.std_dev = static_cast<float>(sd);
    stats.three_sigma = static_cast<float>(3.0 * sd);
    stats.max_abs_err = max_err;
    return stats;
}

}  // namespace wafer_aoi
