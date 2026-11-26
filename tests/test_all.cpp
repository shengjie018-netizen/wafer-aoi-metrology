#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "wafer_aoi/d2d_inspector.hpp"
#include "wafer_aoi/image.hpp"
#include "wafer_aoi/image_ops.hpp"
#include "wafer_aoi/shape_matcher.hpp"
#include "wafer_aoi/subpixel_metrology.hpp"
#include "wafer_aoi/thread_pool.hpp"

using namespace wafer_aoi;

namespace {
int g_passed = 0;
int g_failed = 0;

void check(bool cond, const std::string& name, const std::string& detail = "") {
    if (cond) {
        ++g_passed;
        std::cout << "[PASS] " << name;
        if (!detail.empty()) std::cout << " (" << detail << ")";
        std::cout << "\n";
    } else {
        ++g_failed;
        std::cerr << "[FAIL] " << name;
        if (!detail.empty()) std::cerr << " (" << detail << ")";
        std::cerr << "\n";
    }
}

// Anti-aliased synthetic circle via 4x4 supersampling
ImageF32 render_aa_circle(int w, int h, float cx, float cy, float r,
                          float bg = 40.0f, float fg = 210.0f) {
    ImageF32 img(w, h, 1, bg);
    constexpr int S = 4;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int inside = 0;
            for (int sy = 0; sy < S; ++sy) {
                for (int sx = 0; sx < S; ++sx) {
                    float px = x - 0.5f + (sx + 0.5f) / S;
                    float py = y - 0.5f + (sy + 0.5f) / S;
                    if (std::hypot(px - cx, py - cy) <= r) ++inside;
                }
            }
            float alpha = static_cast<float>(inside) / static_cast<float>(S * S);
            img.at(x, y) = bg + alpha * (fg - bg);
        }
    }
    return img;
}

// Anti-aliased vertical bar for exact pitch measurement
ImageF32 render_aa_bar(int w, int h, float x_left, float x_right,
                       float y_top, float y_bot, float bg = 35.0f, float fg = 215.0f) {
    ImageF32 img(w, h, 1, bg);
    constexpr int S = 4;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int inside = 0;
            for (int sy = 0; sy < S; ++sy) {
                for (int sx = 0; sx < S; ++sx) {
                    float px = x - 0.5f + (sx + 0.5f) / S;
                    float py = y - 0.5f + (sy + 0.5f) / S;
                    if (px >= x_left && px <= x_right && py >= y_top && py <= y_bot) {
                        ++inside;
                    }
                }
            }
            float alpha = static_cast<float>(inside) / static_cast<float>(S * S);
            img.at(x, y) = bg + alpha * (fg - bg);
        }
    }
    return img;
}

ImageF32 render_ic_template(int size = 80) {
    ImageF32 t(size, size, 1, 40.0f);
    // Outer IC package body
    for (int y = 14; y < size - 14; ++y) {
        for (int x = 14; x < size - 14; ++x) t.at(x, y) = 190.0f;
    }
    // Inner dark die cavity
    for (int y = 26; y < size - 26; ++y) {
        for (int x = 26; x < size - 26; ++x) t.at(x, y) = 70.0f;
    }
    // Asymmetric pin-1 fiducial notch (breaks 90/180-deg symmetry!)
    for (int y = 16; y < 32; ++y) {
        for (int x = 16; x < 36; ++x) t.at(x, y) = 240.0f;
    }
    // Pin-1 circular marker
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (std::hypot(x - 55.0f, y - 24.0f) <= 5.5f) t.at(x, y) = 30.0f;
        }
    }
    return gaussian_blur(t, 0.7f);
}

void test_image_ops_and_ccl() {
    ImageU8 bin(64, 64, 1, 0);
    for (int y = 5; y < 15; ++y)
        for (int x = 5; x < 15; ++x) bin.at(x, y) = 255;  // 100 px
    for (int y = 30; y < 35; ++y)
        for (int x = 20; x < 50; ++x) bin.at(x, y) = 255;  // 150 px

    auto comps = connected_components(bin, 10);
    check(comps.size() == 2, "CCL finds 2 components");
    check(comps[0].area == 150 && comps[1].area == 100, "CCL areas exact");

    uint8_t t = otsu_threshold(bin);
    check(t < 255, "Otsu threshold in range");
}

void test_fft_and_phase_correlation() {
    // 1. 1D FFT exact roundtrip
    std::vector<ComplexD> sig(64);
    for (int i = 0; i < 64; ++i) sig[i] = ComplexD(std::sin(2.0 * 3.14159 * i / 8.0) + 0.3 * i, 0.0);
    auto orig = sig;
    fft_1d(sig, false);
    fft_1d(sig, true);
    double max_diff = 0.0;
    for (size_t i = 0; i < 64; ++i) max_diff = std::max(max_diff, std::abs(sig[i] - orig[i]));
    check(max_diff < 1e-10, "1D FFT roundtrip accuracy", "max_err=" + std::to_string(max_diff));

    // 2. Sub-pixel phase correlation on textured wafer image
    const int W = 256, H = 256;
    ImageF32 ref(W, H, 1, 80.0f);
    for (int y = 20; y < H - 20; ++y) {
        for (int x = 20; x < W - 20; ++x) {
            float v = 80.0f;
            if ((x / 16) % 2 == 0) v += 35.0f;
            if ((y / 24) % 2 == 0) v += 35.0f;
            if (x > 90 && x < 145 && y > 70 && y < 130) v = 210.0f;
            if (std::hypot(x - 180.0f, y - 175.0f) < 22.0f) v = 30.0f;
            ref.at(x, y) = v;
        }
    }
    ref = gaussian_blur(ref, 1.0f);

    const float true_dx = 6.35f;
    const float true_dy = -4.72f;
    ImageF32 shifted = shift_subpixel(ref, true_dx, true_dy);
    auto est = phase_correlate_subpixel(ref, shifted);
    float err_x = std::abs(est.dx - true_dx);
    float err_y = std::abs(est.dy - true_dy);
    check(err_x < 0.12f && err_y < 0.12f,
          "Phase correlation sub-pixel shift accuracy",
          "est=(" + std::to_string(est.dx) + "," + std::to_string(est.dy) +
              "), err=(" + std::to_string(err_x) + "," + std::to_string(err_y) + ")");
}

void test_shape_based_matching() {
    ImageF32 templ = render_ic_template(80);
    const int SW = 320, SH = 260;
    const float true_x = 182.4f;
    const float true_y = 136.7f;
    const float true_angle = 34.5f;

    ImageF32 scene = warp_similarity(templ, SW, SH, true_x, true_y, true_angle, 1.0f, 39.5f, 39.5f);
    // Add background + contrast compression + brightness offset + mild noise
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 3.0f);
    for (int y = 0; y < SH; ++y) {
        for (int x = 0; x < SW; ++x) {
            float v = scene.at(x, y);
            if (v < 1.0f) v = 40.0f;
            // Reduce contrast to 70% and shift brightness +25
            scene.at(x, y) = std::clamp(v * 0.70f + 25.0f + noise(rng), 0.0f, 255.0f);
        }
    }

    ShapeMatchConfig cfg;
    cfg.num_levels = 3;
    cfg.angle_min = -180.0f;
    cfg.angle_max = 180.0f;
    cfg.angle_step = 2.0f;
    cfg.grad_thresh = 20.0f;
    cfg.min_score = 0.65f;
    cfg.max_matches = 3;

    ShapeBasedMatcher matcher;
    matcher.train(templ, cfg);
    auto hits = matcher.match(scene);

    check(!hits.empty(), "ShapeBasedMatcher detects rotated & contrast-drifted IC");
    if (!hits.empty()) {
        float ex = std::abs(hits[0].x - true_x);
        float ey = std::abs(hits[0].y - true_y);
        float ea = std::abs(hits[0].angle_deg - true_angle);
        check(ex < 0.8f && ey < 0.8f && ea < 1.0f,
              "ShapeBasedMatcher pose accuracy",
              "est=(" + std::to_string(hits[0].x) + "," + std::to_string(hits[0].y) +
                  "," + std::to_string(hits[0].angle_deg) + " deg, score=" +
                  std::to_string(hits[0].score) + ")");
    }
}

void test_subpixel_metrology() {
    // 1. Sub-pixel circle fit with outlier burr noise
    const float true_cx = 128.42f, true_cy = 127.68f, true_r = 38.35f;
    ImageF32 img = render_aa_circle(256, 256, true_cx, true_cy, true_r);
    // Inject small outlier bump on boundary to test RANSAC robustness
    for (int y = 85; y < 93; ++y)
        for (int x = 124; x < 133; ++x) img.at(x, y) = 210.0f;

    auto edges = extract_subpixel_edges(img, 20.0f, 0.9f);
    auto circle = fit_circle_ransac(edges, 0.4f, 600, 42);
    float err_cx = std::abs(circle.cx - true_cx);
    float err_cy = std::abs(circle.cy - true_cy);
    float err_r = std::abs(circle.radius - true_r);
    check(circle.valid && err_cx < 0.10f && err_cy < 0.10f && err_r < 0.10f,
          "Sub-pixel RANSAC+GN circle metrology (<0.10 px)",
          "err=(" + std::to_string(err_cx) + "," + std::to_string(err_cy) +
              ",r=" + std::to_string(err_r) + ")");

    // 2. Sub-pixel parallel line pitch (IC lead width)
    const float true_pitch = 44.70f;
    ImageF32 bar = render_aa_bar(200, 160, 70.25f, 70.25f + true_pitch, 20.0f, 140.0f);
    auto left_pts = extract_subpixel_edges(bar, 20.0f, 0.9f, 55, 30, 30, 100);
    auto right_pts = extract_subpixel_edges(bar, 20.0f, 0.9f, 100, 30, 30, 100);
    auto l1 = fit_line_ransac(left_pts, 0.4f);
    auto l2 = fit_line_ransac(right_pts, 0.4f);
    float pitch = measure_parallel_line_pitch(l1, l2, 92.0f, 80.0f);
    float pitch_err = std::abs(pitch - true_pitch);
    check(pitch_err < 0.10f,
          "Sub-pixel parallel line pitch metrology (<0.10 px)",
          "est=" + std::to_string(pitch) + ", err=" + std::to_string(pitch_err));
}

void test_fft_notch_and_d2d() {
    // 1. Periodic wafer grating + non-periodic particle
    const int N = 256;
    ImageF32 periodic(N, N, 1, 128.0f);
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            float g = 128.0f + 45.0f * std::sin(2.0f * 3.14159265f * x / 16.0f) +
                      35.0f * std::sin(2.0f * 3.14159265f * y / 16.0f);
            periodic.at(x, y) = g;
        }
    }
    // Inject localized defect
    for (int y = 120; y < 128; ++y)
        for (int x = 130; x < 138; ++x) periodic.at(x, y) = 235.0f;

    auto notch = fft_suppress_periodic_background(periodic, 6.0f, 3.5f, 3);
    // Check background std outside defect dropped drastically
    double var_before = 0.0, var_after = 0.0;
    int cnt = 0;
    for (int y = 20; y < 100; ++y) {
        for (int x = 20; x < 100; ++x) {
            double db = periodic.at(x, y) - 128.0;
            double da = notch.filtered.at(x, y) - 128.0;
            var_before += db * db;
            var_after += da * da;
            ++cnt;
        }
    }
    double ratio = (var_after / cnt) / (var_before / cnt);
    check(notch.num_notches >= 4 && ratio < 0.15,
          "FFT notch filter suppresses periodic wafer background",
          "notches=" + std::to_string(notch.num_notches) + ", residual_var_ratio=" + std::to_string(ratio));

    // 2. Full D2D inspection with sub-pixel shift + brightness drift + 2 defects
    ImageF32 golden(256, 256, 1, 70.0f);
    for (int y = 24; y < 232; ++y) {
        for (int x = 24; x < 232; ++x) {
            if ((x % 32) < 8 || (y % 32) < 8) golden.at(x, y) = 175.0f;
            if (x > 96 && x < 144 && y > 96 && y < 144) golden.at(x, y) = 215.0f;
        }
    }
    golden = gaussian_blur(golden, 0.9f);

    ImageF32 test = golden.clone();
    // Inject Particle (compact) & Scratch (elongated)
    for (int y = 60; y < 68; ++y)
        for (int x = 60; x < 68; ++x) test.at(x, y) = 20.0f;
    for (int i = 0; i < 36; ++i) {
        for (int t = -1; t <= 1; ++t) test.at(160 + i, 180 + t) = 245.0f;
    }
    // Shift test by sub-pixel (+3.4, -2.6) and drift brightness
    test = shift_subpixel(test, 3.4f, -2.6f);
    for (size_t i = 0; i < test.size(); ++i) {
        test.data()[i] = std::clamp(test.data()[i] * 1.10f + 10.0f, 0.0f, 255.0f);
    }

    D2DConfig dcfg;
    dcfg.base_diff_thresh = 20.0f;
    dcfg.edge_tolerance_k = 0.18f;
    dcfg.min_defect_area = 10;
    auto d2d = inspect_die_to_die(golden, test, dcfg);
    check(d2d.defects.size() == 2,
          "D2D detects exact 2 defects with 0 overkill under shift+illumination drift",
          "found=" + std::to_string(d2d.defects.size()) +
              ", shift=(" + std::to_string(d2d.alignment.dx) + "," +
              std::to_string(d2d.alignment.dy) + ")");
}

void test_thread_pool() {
    ThreadPool pool(4);
    std::vector<std::future<int>> futs;
    for (int i = 1; i <= 16; ++i) {
        futs.push_back(pool.enqueue([i]() { return i * i; }));
    }
    int sum = 0;
    for (auto& f : futs) sum += f.get();
    check(sum == 1496, "ThreadPool parallel execution sum exact");
}
}  // namespace

int main() {
    std::cout << "=== Running Wafer AOI Metrology Unit Tests ===\n";
    test_image_ops_and_ccl();
    test_fft_and_phase_correlation();
    test_shape_based_matching();
    test_subpixel_metrology();
    test_fft_notch_and_d2d();
    test_thread_pool();
    std::cout << "=== Summary: " << g_passed << " passed, " << g_failed << " failed ===\n";
    return (g_failed == 0) ? 0 : 1;
}
