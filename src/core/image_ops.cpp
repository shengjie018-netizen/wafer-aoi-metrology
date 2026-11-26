#include "wafer_aoi/image_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace wafer_aoi {

static inline int reflect_index(int i, int n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

ImageF32 gaussian_blur(const ImageF32& src, float sigma) {
    if (src.empty() || sigma <= 1e-3f) return src.clone();

    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    const int ksize = 2 * radius + 1;
    std::vector<float> kernel(ksize);
    float sum = 0.0f;
    const float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);
    for (int i = -radius; i <= radius; ++i) {
        float w = std::exp(-static_cast<float>(i * i) * inv_2sigma2);
        kernel[i + radius] = w;
        sum += w;
    }
    for (float& w : kernel) w /= sum;

    const int w = src.width();
    const int h = src.height();
    ImageF32 temp(w, h, 1);
    ImageF32 dst(w, h, 1);

    // Horizontal pass
#ifdef WAFER_AOI_OPENMP
#pragma omp parallel for schedule(static) if (h >= 128)
#endif
    for (int y = 0; y < h; ++y) {
        const float* srow = src.row_ptr(y);
        float* trow = temp.row_ptr(y);
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sx = reflect_index(x + k, w);
                acc += srow[sx] * kernel[k + radius];
            }
            trow[x] = acc;
        }
    }

    // Vertical pass
#ifdef WAFER_AOI_OPENMP
#pragma omp parallel for schedule(static) if (h >= 128)
#endif
    for (int y = 0; y < h; ++y) {
        float* drow = dst.row_ptr(y);
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sy = reflect_index(y + k, h);
                acc += temp.at(x, sy) * kernel[k + radius];
            }
            drow[x] = acc;
        }
    }
    return dst;
}

ImageF32 pyr_down(const ImageF32& src) {
    ImageF32 smooth = gaussian_blur(src, 1.0f);
    const int nw = std::max(1, src.width() / 2);
    const int nh = std::max(1, src.height() / 2);
    ImageF32 dst(nw, nh, 1);
    for (int y = 0; y < nh; ++y) {
        const float* srow = smooth.row_ptr(std::min(2 * y, src.height() - 1));
        float* drow = dst.row_ptr(y);
        for (int x = 0; x < nw; ++x) {
            drow[x] = srow[std::min(2 * x, src.width() - 1)];
        }
    }
    return dst;
}

std::vector<ImageF32> build_pyramid(const ImageF32& src, int num_levels) {
    std::vector<ImageF32> pyr;
    pyr.reserve(num_levels);
    pyr.push_back(src.clone());
    for (int l = 1; l < num_levels; ++l) {
        pyr.push_back(pyr_down(pyr.back()));
    }
    return pyr;
}

void sobel_gradients(const ImageF32& src, ImageF32& gx, ImageF32& gy, ImageF32& mag) {
    const int w = src.width();
    const int h = src.height();
    gx = ImageF32(w, h, 1, 0.0f);
    gy = ImageF32(w, h, 1, 0.0f);
    mag = ImageF32(w, h, 1, 0.0f);

#ifdef WAFER_AOI_OPENMP
#pragma omp parallel for schedule(static) if (h >= 128)
#endif
    for (int y = 1; y < h - 1; ++y) {
        const float* r0 = src.row_ptr(y - 1);
        const float* r1 = src.row_ptr(y);
        const float* r2 = src.row_ptr(y + 1);
        float* gx_row = gx.row_ptr(y);
        float* gy_row = gy.row_ptr(y);
        float* m_row = mag.row_ptr(y);
        for (int x = 1; x < w - 1; ++x) {
            // Normalized by 4 so a unit step edge of height H yields ~H magnitude
            float dx = ((-r0[x - 1] + r0[x + 1]) +
                        2.0f * (-r1[x - 1] + r1[x + 1]) +
                        (-r2[x - 1] + r2[x + 1])) * 0.25f;
            float dy = ((-r0[x - 1] - 2.0f * r0[x] - r0[x + 1]) +
                        (r2[x - 1] + 2.0f * r2[x] + r2[x + 1])) * 0.25f;
            gx_row[x] = dx;
            gy_row[x] = dy;
            m_row[x] = std::sqrt(dx * dx + dy * dy);
        }
    }
}

float bilinear_sample(const ImageF32& img, float x, float y) {
    const int w = img.width();
    const int h = img.height();
    if (w == 0 || h == 0) return 0.0f;
    x = std::clamp(x, 0.0f, static_cast<float>(w - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(h - 1));
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, w - 1);
    int y1 = std::min(y0 + 1, h - 1);
    float fx = x - static_cast<float>(x0);
    float fy = y - static_cast<float>(y0);

    float v00 = img.at(x0, y0);
    float v10 = img.at(x1, y0);
    float v01 = img.at(x0, y1);
    float v11 = img.at(x1, y1);
    float top = v00 + fx * (v10 - v00);
    float bot = v01 + fx * (v11 - v01);
    return top + fy * (bot - top);
}

ImageF32 shift_subpixel(const ImageF32& src, float dx, float dy) {
    const int w = src.width();
    const int h = src.height();
    ImageF32 dst(w, h, 1);
#ifdef WAFER_AOI_OPENMP
#pragma omp parallel for schedule(static) if (h >= 128)
#endif
    for (int y = 0; y < h; ++y) {
        float* drow = dst.row_ptr(y);
        float sy = static_cast<float>(y) - dy;
        for (int x = 0; x < w; ++x) {
            drow[x] = bilinear_sample(src, static_cast<float>(x) - dx, sy);
        }
    }
    return dst;
}

ImageF32 warp_similarity(const ImageF32& src, int dst_w, int dst_h,
                         float tx, float ty, float angle_deg, float scale,
                         float cx, float cy) {
    ImageF32 dst(dst_w, dst_h, 1, 0.0f);
    const float rad = angle_deg * 3.14159265358979323846f / 180.0f;
    const float cos_a = std::cos(rad);
    const float sin_a = std::sin(rad);
    const float inv_s = (std::abs(scale) > 1e-6f) ? (1.0f / scale) : 1.0f;

#ifdef WAFER_AOI_OPENMP
#pragma omp parallel for schedule(static) if (dst_h >= 128)
#endif
    for (int y = 0; y < dst_h; ++y) {
        float* drow = dst.row_ptr(y);
        float dy = static_cast<float>(y) - ty;
        for (int x = 0; x < dst_w; ++x) {
            float dx = static_cast<float>(x) - tx;
            // Inverse transform: map dst -> src
            float sx = (cos_a * dx + sin_a * dy) * inv_s + cx;
            float sy = (-sin_a * dx + cos_a * dy) * inv_s + cy;
            if (sx >= 0.0f && sy >= 0.0f && sx <= src.width() - 1 && sy <= src.height() - 1) {
                drow[x] = bilinear_sample(src, sx, sy);
            }
        }
    }
    return dst;
}

std::vector<double> compute_integral_image(const ImageF32& src) {
    const int w = src.width();
    const int h = src.height();
    std::vector<double> integ(static_cast<size_t>(w + 1) * static_cast<size_t>(h + 1), 0.0);
    for (int y = 1; y <= h; ++y) {
        double row_sum = 0.0;
        const float* srow = src.row_ptr(y - 1);
        for (int x = 1; x <= w; ++x) {
            row_sum += static_cast<double>(srow[x - 1]);
            integ[static_cast<size_t>(y) * (w + 1) + x] =
                integ[static_cast<size_t>(y - 1) * (w + 1) + x] + row_sum;
        }
    }
    return integ;
}

uint8_t otsu_threshold(const ImageU8& src) {
    std::array<uint64_t, 256> hist{};
    for (size_t i = 0; i < src.size(); ++i) {
        hist[src.data()[i]]++;
    }
    const double total = static_cast<double>(src.size());
    if (total <= 0.0) return 0;

    double sum_total = 0.0;
    for (int i = 0; i < 256; ++i) sum_total += i * static_cast<double>(hist[i]);

    double sum_b = 0.0;
    double w_b = 0.0;
    double best_var = -1.0;
    int best_t = 0;

    for (int t = 0; t < 256; ++t) {
        w_b += static_cast<double>(hist[t]);
        if (w_b <= 0.0) continue;
        double w_f = total - w_b;
        if (w_f <= 0.0) break;
        sum_b += t * static_cast<double>(hist[t]);
        double m_b = sum_b / w_b;
        double m_f = (sum_total - sum_b) / w_f;
        double diff = m_b - m_f;
        double var_between = w_b * w_f * diff * diff;
        if (var_between > best_var) {
            best_var = var_between;
            best_t = t;
        }
    }
    return static_cast<uint8_t>(best_t);
}

ImageU8 morph_erode(const ImageU8& binary, int radius) {
    if (radius <= 0) return binary.clone();
    const int w = binary.width();
    const int h = binary.height();
    ImageU8 dst(w, h, 1, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t mn = 255;
            for (int dy = -radius; dy <= radius && mn > 0; ++dy) {
                int sy = std::clamp(y + dy, 0, h - 1);
                const uint8_t* row = binary.row_ptr(sy);
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = std::clamp(x + dx, 0, w - 1);
                    if (row[sx] < mn) mn = row[sx];
                    if (mn == 0) break;
                }
            }
            dst.at(x, y) = mn;
        }
    }
    return dst;
}

ImageU8 morph_dilate(const ImageU8& binary, int radius) {
    if (radius <= 0) return binary.clone();
    const int w = binary.width();
    const int h = binary.height();
    ImageU8 dst(w, h, 1, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t mx = 0;
            for (int dy = -radius; dy <= radius && mx < 255; ++dy) {
                int sy = std::clamp(y + dy, 0, h - 1);
                const uint8_t* row = binary.row_ptr(sy);
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = std::clamp(x + dx, 0, w - 1);
                    if (row[sx] > mx) mx = row[sx];
                    if (mx == 255) break;
                }
            }
            dst.at(x, y) = mx;
        }
    }
    return dst;
}

ImageU8 morph_open(const ImageU8& binary, int radius) {
    return morph_dilate(morph_erode(binary, radius), radius);
}

ImageU8 morph_close(const ImageU8& binary, int radius) {
    return morph_erode(morph_dilate(binary, radius), radius);
}

ImageF32 morph_tophat_gray(const ImageF32& src, int radius) {
    if (radius <= 0) return ImageF32(src.width(), src.height(), 1, 0.0f);
    const int w = src.width();
    const int h = src.height();
    ImageF32 eroded(w, h, 1);
    ImageF32 opened(w, h, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float mn = std::numeric_limits<float>::max();
            for (int dy = -radius; dy <= radius; ++dy) {
                int sy = std::clamp(y + dy, 0, h - 1);
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = std::clamp(x + dx, 0, w - 1);
                    mn = std::min(mn, src.at(sx, sy));
                }
            }
            eroded.at(x, y) = mn;
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float mx = std::numeric_limits<float>::lowest();
            for (int dy = -radius; dy <= radius; ++dy) {
                int sy = std::clamp(y + dy, 0, h - 1);
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = std::clamp(x + dx, 0, w - 1);
                    mx = std::max(mx, eroded.at(sx, sy));
                }
            }
            opened.at(x, y) = mx;
        }
    }
    ImageF32 out(w, h, 1);
    for (size_t i = 0; i < src.size(); ++i) {
        out.data()[i] = std::max(0.0f, src.data()[i] - opened.data()[i]);
    }
    return out;
}

namespace {
struct DisjointSet {
    std::vector<int> parent;
    int make_set() {
        int id = static_cast<int>(parent.size());
        parent.push_back(id);
        return id;
    }
    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) {
            if (a < b) parent[b] = a;
            else parent[a] = b;
        }
    }
};
}  // namespace

std::vector<ConnectedComponent> connected_components(
    const ImageU8& binary, int min_area, ImageI32* labels_out) {
    const int w = binary.width();
    const int h = binary.height();
    ImageI32 labels(w, h, 1, 0);
    DisjointSet uf;
    uf.make_set();  // 0 = background

    // First pass: assign provisional labels and record equivalences (8-connectivity)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (binary.at(x, y) == 0) continue;
            int neighbors[4] = {0, 0, 0, 0};
            int count = 0;
            if (x > 0 && labels.at(x - 1, y) > 0) neighbors[count++] = labels.at(x - 1, y);
            if (y > 0 && x > 0 && labels.at(x - 1, y - 1) > 0) neighbors[count++] = labels.at(x - 1, y - 1);
            if (y > 0 && labels.at(x, y - 1) > 0) neighbors[count++] = labels.at(x, y - 1);
            if (y > 0 && x + 1 < w && labels.at(x + 1, y - 1) > 0) neighbors[count++] = labels.at(x + 1, y - 1);

            if (count == 0) {
                labels.at(x, y) = uf.make_set();
            } else {
                int min_lbl = neighbors[0];
                for (int i = 1; i < count; ++i) min_lbl = std::min(min_lbl, neighbors[i]);
                labels.at(x, y) = min_lbl;
                for (int i = 0; i < count; ++i) uf.unite(min_lbl, neighbors[i]);
            }
        }
    }

    // Second pass: resolve roots & accumulate component stats
    struct Acc {
        int area = 0;
        int min_x = std::numeric_limits<int>::max();
        int min_y = std::numeric_limits<int>::max();
        int max_x = -1;
        int max_y = -1;
        double sum_x = 0.0;
        double sum_y = 0.0;
    };
    std::unordered_map<int, Acc> stats;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int lbl = labels.at(x, y);
            if (lbl == 0) continue;
            int root = uf.find(lbl);
            labels.at(x, y) = root;
            Acc& a = stats[root];
            a.area++;
            a.min_x = std::min(a.min_x, x);
            a.min_y = std::min(a.min_y, y);
            a.max_x = std::max(a.max_x, x);
            a.max_y = std::max(a.max_y, y);
            a.sum_x += x;
            a.sum_y += y;
        }
    }

    std::vector<ConnectedComponent> comps;
    std::unordered_map<int, int> root_to_id;
    int next_id = 1;
    for (const auto& [root, a] : stats) {
        if (a.area < min_area) continue;
        ConnectedComponent c;
        c.id = next_id++;
        root_to_id[root] = c.id;
        c.area = a.area;
        c.x = a.min_x;
        c.y = a.min_y;
        c.w = a.max_x - a.min_x + 1;
        c.h = a.max_y - a.min_y + 1;
        c.cx = static_cast<float>(a.sum_x / a.area);
        c.cy = static_cast<float>(a.sum_y / a.area);
        float mn_dim = static_cast<float>(std::min(c.w, c.h));
        float mx_dim = static_cast<float>(std::max(c.w, c.h));
        c.aspect_ratio = mx_dim / std::max(1.0f, mn_dim);
        c.fill_ratio = static_cast<float>(c.area) / static_cast<float>(c.w * c.h);
        comps.push_back(c);
    }

    std::sort(comps.begin(), comps.end(),
              [](const ConnectedComponent& a, const ConnectedComponent& b) {
                  return a.area > b.area;
              });

    if (labels_out) {
        for (size_t i = 0; i < labels.size(); ++i) {
            int root = labels.data()[i];
            auto it = root_to_id.find(root);
            labels.data()[i] = (it != root_to_id.end()) ? it->second : 0;
        }
        *labels_out = std::move(labels);
    }
    return comps;
}

}  // namespace wafer_aoi
