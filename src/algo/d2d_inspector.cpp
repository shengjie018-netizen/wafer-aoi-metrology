#include "wafer_aoi/d2d_inspector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace wafer_aoi {

namespace {
constexpr double kPiD = 3.14159265358979323846;

int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

ImageF32 fft_shift_mag(const std::vector<ComplexD>& freq, int pw, int ph) {
    ImageF32 log_mag(pw, ph, 1, 0.0f);
    int hw = pw / 2;
    int hh = ph / 2;
    for (int y = 0; y < ph; ++y) {
        int sy = (y + hh) % ph;
        for (int x = 0; x < pw; ++x) {
            int sx = (x + hw) % pw;
            double mag = std::abs(freq[static_cast<size_t>(y) * pw + x]);
            log_mag.at(sx, sy) = static_cast<float>(std::log1p(mag));
        }
    }
    return log_mag;
}
}  // namespace

void fft_1d(std::vector<ComplexD>& a, bool inverse) {
    const size_t n = a.size();
    if (n <= 1) return;
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("fft_1d requires power-of-two length");
    }

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // Iterative Cooley-Tukey butterflies
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * kPiD / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        ComplexD wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            ComplexD w(1.0, 0.0);
            size_t half = len >> 1;
            for (size_t j = 0; j < half; ++j) {
                ComplexD u = a[i + j];
                ComplexD v = a[i + j + half] * w;
                a[i + j] = u + v;
                a[i + j + half] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto& x : a) x *= inv_n;
    }
}

void fft_2d(std::vector<ComplexD>& data, int width, int height, bool inverse) {
    if (static_cast<int>(data.size()) != width * height) {
        throw std::invalid_argument("fft_2d size mismatch");
    }

    // 1D FFT on all rows
#ifdef WAFER_AOI_OPENMP
#pragma omp parallel for schedule(static) if (height >= 64)
#endif
    for (int y = 0; y < height; ++y) {
        std::vector<ComplexD> row(width);
        const size_t offset = static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) row[x] = data[offset + x];
        fft_1d(row, inverse);
        for (int x = 0; x < width; ++x) data[offset + x] = row[x];
    }

    // 1D FFT on all columns
#ifdef WAFER_AOI_OPENMP
#pragma omp parallel for schedule(static) if (width >= 64)
#endif
    for (int x = 0; x < width; ++x) {
        std::vector<ComplexD> col(height);
        for (int y = 0; y < height; ++y) col[y] = data[static_cast<size_t>(y) * width + x];
        fft_1d(col, inverse);
        for (int y = 0; y < height; ++y) data[static_cast<size_t>(y) * width + x] = col[y];
    }
}

PhaseShiftResult phase_correlate_subpixel(const ImageF32& ref, const ImageF32& test) {
    PhaseShiftResult res;
    if (ref.empty() || test.empty() ||
        ref.width() != test.width() || ref.height() != test.height()) {
        return res;
    }

    const int w = ref.width();
    const int h = ref.height();
    const int pw = next_pow2(w);
    const int ph = next_pow2(h);

    // Compute means to remove DC leakage before Hann windowing
    double mean_r = 0.0, mean_t = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        mean_r += ref.data()[i];
        mean_t += test.data()[i];
    }
    mean_r /= static_cast<double>(ref.size());
    mean_t /= static_cast<double>(test.size());

    std::vector<double> wx(w), wy(h);
    for (int x = 0; x < w; ++x) {
        wx[x] = (w > 1) ? 0.5 * (1.0 - std::cos(2.0 * kPiD * x / (w - 1))) : 1.0;
    }
    for (int y = 0; y < h; ++y) {
        wy[y] = (h > 1) ? 0.5 * (1.0 - std::cos(2.0 * kPiD * y / (h - 1))) : 1.0;
    }

    std::vector<ComplexD> F_ref(static_cast<size_t>(pw) * ph, ComplexD(0.0, 0.0));
    std::vector<ComplexD> F_test(static_cast<size_t>(pw) * ph, ComplexD(0.0, 0.0));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double win = wx[x] * wy[y];
            size_t idx = static_cast<size_t>(y) * pw + x;
            F_ref[idx] = ComplexD((ref.at(x, y) - mean_r) * win, 0.0);
            F_test[idx] = ComplexD((test.at(x, y) - mean_t) * win, 0.0);
        }
    }

    fft_2d(F_ref, pw, ph, false);
    fft_2d(F_test, pw, ph, false);

    // Regularized cross-power spectrum: F_test * conj(F_ref) / (|...| + eps)
    // Regularizing by a small fraction of peak magnitude prevents amplifying
    // near-zero Nyquist noise / bilinear-interpolation artifacts.
    std::vector<ComplexD> cps_freq(static_cast<size_t>(pw) * ph);
    double max_mag = 0.0;
    for (size_t i = 0; i < cps_freq.size(); ++i) {
        cps_freq[i] = F_test[i] * std::conj(F_ref[i]);
        max_mag = std::max(max_mag, std::abs(cps_freq[i]));
    }
    const double eps = std::max(1e-12, max_mag * 1e-4);
    for (size_t i = 0; i < cps_freq.size(); ++i) {
        double mag = std::abs(cps_freq[i]);
        cps_freq[i] = cps_freq[i] / (mag + eps);
    }

    std::vector<ComplexD> corr = cps_freq;
    fft_2d(corr, pw, ph, true);

    // Locate integer correlation peak
    int peak_x = 0, peak_y = 0;
    double best_val = -1e30;
    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            double v = corr[static_cast<size_t>(y) * pw + x].real();
            if (v > best_val) {
                best_val = v;
                peak_x = x;
                peak_y = y;
            }
        }
    }

    // Sub-pixel Guizar-Sicairos (2008) Matrix-Multiply Upsampled DFT refinement
    // Evaluates exact band-limited inverse DFT on a 21x21 grid spanning [-0.5, +0.5] px
    // around (peak_x, peak_y) at 0.05 px step, followed by fine parabolic interpolation.
    constexpr int kHalf = 10;
    constexpr int kGrid = 2 * kHalf + 1;  // 21
    constexpr double kStep = 0.05;

    std::vector<std::vector<ComplexD>> exp_x(kGrid, std::vector<ComplexD>(pw));
    std::vector<std::vector<ComplexD>> exp_y(kGrid, std::vector<ComplexD>(ph));
    for (int m = 0; m < kGrid; ++m) {
        double xq = static_cast<double>(peak_x) + (m - kHalf) * kStep;
        double yq = static_cast<double>(peak_y) + (m - kHalf) * kStep;
        for (int u = 0; u < pw; ++u) {
            int ku = (u < pw / 2) ? u : (u - pw);
            double ang = 2.0 * kPiD * static_cast<double>(ku) * xq / static_cast<double>(pw);
            exp_x[m][u] = ComplexD(std::cos(ang), std::sin(ang));
        }
        for (int v = 0; v < ph; ++v) {
            int kv = (v < ph / 2) ? v : (v - ph);
            double ang = 2.0 * kPiD * static_cast<double>(kv) * yq / static_cast<double>(ph);
            exp_y[m][v] = ComplexD(std::cos(ang), std::sin(ang));
        }
    }

    // Separable 1D contraction along u: tmp[v][m] = sum_u cps_freq[v, u] * exp_x[m][u]
    std::vector<ComplexD> tmp(static_cast<size_t>(ph) * kGrid, ComplexD(0.0, 0.0));
#ifdef WAFER_AOI_OPENMP
#pragma omp parallel for schedule(static) if (ph >= 64)
#endif
    for (int v = 0; v < ph; ++v) {
        const ComplexD* row = cps_freq.data() + static_cast<size_t>(v) * pw;
        for (int m = 0; m < kGrid; ++m) {
            const ComplexD* ex = exp_x[m].data();
            ComplexD acc(0.0, 0.0);
            for (int u = 0; u < pw; ++u) {
                acc += row[u] * ex[u];
            }
            tmp[static_cast<size_t>(v) * kGrid + m] = acc;
        }
    }

    // Contraction along v to get 21x21 upsampled correlation patch
    double patch[kGrid][kGrid];
    int best_m = kHalf, best_n = kHalf;
    double best_up = -1e30;
    for (int n = 0; n < kGrid; ++n) {
        const ComplexD* ey = exp_y[n].data();
        for (int m = 0; m < kGrid; ++m) {
            ComplexD acc(0.0, 0.0);
            for (int v = 0; v < ph; ++v) {
                acc += tmp[static_cast<size_t>(v) * kGrid + m] * ey[v];
            }
            double val = acc.real();
            patch[n][m] = val;
            if (val > best_up) {
                best_up = val;
                best_m = m;
                best_n = n;
            }
        }
    }

    double fine_dx = (best_m - kHalf) * kStep;
    double fine_dy = (best_n - kHalf) * kStep;
    if (best_m > 0 && best_m < kGrid - 1) {
        double d = patch[best_n][best_m - 1] - 2.0 * patch[best_n][best_m] + patch[best_n][best_m + 1];
        if (d < -1e-12) {
            fine_dx += kStep * std::clamp(0.5 * (patch[best_n][best_m - 1] - patch[best_n][best_m + 1]) / d, -0.5, 0.5);
        }
    }
    if (best_n > 0 && best_n < kGrid - 1) {
        double d = patch[best_n - 1][best_m] - 2.0 * patch[best_n][best_m] + patch[best_n + 1][best_m];
        if (d < -1e-12) {
            fine_dy += kStep * std::clamp(0.5 * (patch[best_n - 1][best_m] - patch[best_n + 1][best_m]) / d, -0.5, 0.5);
        }
    }

    double sub_x = fine_dx;
    double sub_y = fine_dy;

    double dx = static_cast<double>(peak_x) + sub_x;
    double dy = static_cast<double>(peak_y) + sub_y;
    if (dx > pw * 0.5) dx -= pw;
    if (dy > ph * 0.5) dy -= ph;

    res.dx = static_cast<float>(dx);
    res.dy = static_cast<float>(dy);
    res.peak_value = static_cast<float>(best_val);
    return res;
}

FFTNotchResult fft_suppress_periodic_background(
    const ImageF32& src, float dc_keep_radius, float spike_zscore, int notch_radius) {
    FFTNotchResult out;
    if (src.empty()) return out;

    const int w = src.width();
    const int h = src.height();
    const int pw = next_pow2(w);
    const int ph = next_pow2(h);

    double mean = 0.0;
    for (size_t i = 0; i < src.size(); ++i) mean += src.data()[i];
    mean /= static_cast<double>(src.size());

    std::vector<ComplexD> freq(static_cast<size_t>(pw) * ph, ComplexD(0.0, 0.0));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            freq[static_cast<size_t>(y) * pw + x] = ComplexD(src.at(x, y) - mean, 0.0);
        }
    }

    fft_2d(freq, pw, ph, false);
    out.log_spectrum_before = fft_shift_mag(freq, pw, ph);

    // Radial-ring statistics to robustly identify sharp periodic peaks
    const int hw = pw / 2;
    const int hh = ph / 2;
    const int max_r = static_cast<int>(std::ceil(std::hypot(hw, hh))) + 2;
    std::vector<double> ring_sum(max_r, 0.0), ring_sq(max_r, 0.0);
    std::vector<int> ring_cnt(max_r, 0);

    for (int sy = 0; sy < ph; ++sy) {
        for (int sx = 0; sx < pw; ++sx) {
            float r = std::hypot(static_cast<float>(sx - hw), static_cast<float>(sy - hh));
            int ri = std::min(max_r - 1, static_cast<int>(std::lround(r)));
            double v = out.log_spectrum_before.at(sx, sy);
            ring_sum[ri] += v;
            ring_sq[ri] += v * v;
            ring_cnt[ri]++;
        }
    }

    std::vector<double> ring_mean(max_r, 0.0), ring_std(max_r, 1.0);
    for (int r = 0; r < max_r; ++r) {
        if (ring_cnt[r] > 1) {
            ring_mean[r] = ring_sum[r] / ring_cnt[r];
            double var = ring_sq[r] / ring_cnt[r] - ring_mean[r] * ring_mean[r];
            ring_std[r] = std::sqrt(std::max(1e-6, var));
        }
    }

    // Build smooth Gaussian notch mask in shifted frequency coordinates
    ImageF32 mask(pw, ph, 1, 1.0f);
    std::vector<std::pair<int, int>> spikes;

    for (int sy = 1; sy < ph - 1; ++sy) {
        for (int sx = 1; sx < pw - 1; ++sx) {
            float r = std::hypot(static_cast<float>(sx - hw), static_cast<float>(sy - hh));
            if (r < dc_keep_radius) continue;
            int ri = std::min(max_r - 1, static_cast<int>(std::lround(r)));
            float v = out.log_spectrum_before.at(sx, sy);
            double z = (v - ring_mean[ri]) / ring_std[ri];
            if (z < spike_zscore) continue;

            // Local maximum check in 3x3 neighborhood
            bool is_peak = true;
            for (int dy = -1; dy <= 1 && is_peak; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (out.log_spectrum_before.at(sx + dx, sy + dy) > v) {
                        is_peak = false;
                        break;
                    }
                }
            }
            if (is_peak) spikes.emplace_back(sx, sy);
        }
    }

    out.num_notches = static_cast<int>(spikes.size());
    const float sigma_n = std::max(1.5f, static_cast<float>(notch_radius) * 0.65f);
    const float inv_2s2 = 1.0f / (2.0f * sigma_n * sigma_n);
    const int rad = std::max(2, notch_radius * 2);

    for (const auto& [ux, uy] : spikes) {
        for (int dy = -rad; dy <= rad; ++dy) {
            int y = uy + dy;
            if (y < 0 || y >= ph) continue;
            for (int dx = -rad; dx <= rad; ++dx) {
                int x = ux + dx;
                if (x < 0 || x >= pw) continue;
                float d2 = static_cast<float>(dx * dx + dy * dy);
                float atten = 1.0f - std::exp(-d2 * inv_2s2);
                mask.at(x, y) *= atten;
            }
        }
    }

    // Apply notch mask (unshift back to native FFT layout)
    for (int y = 0; y < ph; ++y) {
        int sy = (y + hh) % ph;
        for (int x = 0; x < pw; ++x) {
            int sx = (x + hw) % pw;
            freq[static_cast<size_t>(y) * pw + x] *= static_cast<double>(mask.at(sx, sy));
        }
    }

    out.log_spectrum_after = fft_shift_mag(freq, pw, ph);
    fft_2d(freq, pw, ph, true);

    out.filtered = ImageF32(w, h, 1);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            out.filtered.at(x, y) =
                static_cast<float>(freq[static_cast<size_t>(y) * pw + x].real() + mean);
        }
    }
    return out;
}

D2DInspectionResult inspect_die_to_die(
    const ImageF32& golden, const ImageF32& test, const D2DConfig& cfg) {
    auto t0 = std::chrono::steady_clock::now();
    D2DInspectionResult result;
    if (golden.empty() || test.empty() ||
        golden.width() != test.width() || golden.height() != test.height()) {
        return result;
    }

    const int w = golden.width();
    const int h = golden.height();

    // 1. Sub-pixel Die-to-Die alignment via Phase Correlation
    result.alignment = phase_correlate_subpixel(golden, test);
    result.aligned_test = shift_subpixel(test, -result.alignment.dx, -result.alignment.dy);

    // 2. Illumination gain & offset normalization (handles lamp / exposure drift)
    int b = std::min(cfg.border_ignore, std::min(w, h) / 4);
    if (cfg.normalize_illumination) {
        double sum_g = 0.0, sum_t = 0.0;
        double sq_g = 0.0, sq_t = 0.0;
        int count = 0;
        for (int y = b; y < h - b; ++y) {
            for (int x = b; x < w - b; ++x) {
                double gv = golden.at(x, y);
                double tv = result.aligned_test.at(x, y);
                sum_g += gv;
                sum_t += tv;
                sq_g += gv * gv;
                sq_t += tv * tv;
                ++count;
            }
        }
        if (count > 10) {
            double mg = sum_g / count;
            double mt = sum_t / count;
            double sg = std::sqrt(std::max(1e-6, sq_g / count - mg * mg));
            double st = std::sqrt(std::max(1e-6, sq_t / count - mt * mt));
            double gain = std::clamp(sg / st, 0.5, 2.0);
            for (size_t i = 0; i < result.aligned_test.size(); ++i) {
                float v = static_cast<float>((result.aligned_test.data()[i] - mt) * gain + mg);
                result.aligned_test.data()[i] = std::clamp(v, 0.0f, 255.0f);
            }
        }
    }

    // 3. Gradient-adaptive residual thresholding (suppresses edge-halo overkill)
    ImageF32 gx, gy, gmag;
    sobel_gradients(golden, gx, gy, gmag);

    result.residual_map = ImageF32(w, h, 1, 0.0f);
    ImageU8 raw_mask(w, h, 1, 0);

    for (int y = b; y < h - b; ++y) {
        for (int x = b; x < w - b; ++x) {
            float diff = std::abs(golden.at(x, y) - result.aligned_test.at(x, y));
            result.residual_map.at(x, y) = diff;
            float local_thresh = cfg.base_diff_thresh + cfg.edge_tolerance_k * gmag.at(x, y);
            if (diff > local_thresh) {
                raw_mask.at(x, y) = 255;
            }
        }
    }

    // 4. Morphological denoising & scratch fragment bridging
    ImageU8 clean = morph_open(raw_mask, cfg.morph_open_radius);
    result.defect_mask = morph_close(clean, cfg.morph_close_radius);

    // 5. Connected Component Labeling & rule-based defect classification
    ImageI32 labels;
    auto comps = connected_components(result.defect_mask, cfg.min_defect_area, &labels);
    result.defects.reserve(comps.size());

    for (const auto& c : comps) {
        float max_sev = 0.0f;
        double mu20 = 0.0, mu02 = 0.0, mu11 = 0.0;
        for (int y = c.y; y < c.y + c.h; ++y) {
            for (int x = c.x; x < c.x + c.w; ++x) {
                if (labels.at(x, y) == c.id) {
                    max_sev = std::max(max_sev, result.residual_map.at(x, y));
                    double dx = x - c.cx;
                    double dy = y - c.cy;
                    mu20 += dx * dx;
                    mu02 += dy * dy;
                    mu11 += dx * dy;
                }
            }
        }
        // Rotation-invariant PCA elongation from 2nd central moments
        double tr = mu20 + mu02;
        double diff_m = mu20 - mu02;
        double disc = std::sqrt(std::max(0.0, diff_m * diff_m + 4.0 * mu11 * mu11));
        double lam1 = std::max(1e-6, 0.5 * (tr + disc));
        double lam2 = std::max(1e-6, 0.5 * (tr - disc));
        float pca_elongation = static_cast<float>(std::sqrt(lam1 / lam2));

        DefectInfo d;
        d.id = c.id;
        d.x = c.x;
        d.y = c.y;
        d.w = c.w;
        d.h = c.h;
        d.area = c.area;
        d.cx = c.cx;
        d.cy = c.cy;
        d.max_severity = max_sev;
        d.aspect_ratio = pca_elongation;
        if (pca_elongation >= 3.0f) {
            d.type = "Scratch";
        } else if (c.area >= 350) {
            d.type = "Missing/LargeDefect";
        } else {
            d.type = "Particle";
        }
        result.defects.push_back(std::move(d));
    }

    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

}  // namespace wafer_aoi
