#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "wafer_aoi/d2d_inspector.hpp"
#include "wafer_aoi/image.hpp"
#include "wafer_aoi/image_ops.hpp"
#include "wafer_aoi/shape_matcher.hpp"
#include "wafer_aoi/subpixel_metrology.hpp"
#include "wafer_aoi/thread_pool.hpp"

using namespace wafer_aoi;
namespace fs = std::filesystem;

namespace {
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

void draw_rotated_box(ImageU8& rgb, float cx, float cy, float w, float h,
                      float angle_deg, uint8_t r, uint8_t g, uint8_t b) {
    float rad = angle_deg * 3.14159265358979323846f / 180.0f;
    float c = std::cos(rad), s = std::sin(rad);
    float hw = w * 0.5f, hh = h * 0.5f;
    float xs[4] = {-hw, hw, hw, -hw};
    float ys[4] = {-hh, -hh, hh, hh};
    int px[4], py[4];
    for (int i = 0; i < 4; ++i) {
        px[i] = static_cast<int>(std::lround(cx + xs[i] * c - ys[i] * s));
        py[i] = static_cast<int>(std::lround(cy + xs[i] * s + ys[i] * c));
    }
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        draw_line(rgb, px[i], py[i], px[j], py[j], r, g, b);
    }
    // Draw orientation arrow from center toward +X axis of template
    int ax = static_cast<int>(std::lround(cx + hw * 0.65f * c));
    int ay = static_cast<int>(std::lround(cy + hw * 0.65f * s));
    draw_line(rgb, static_cast<int>(std::lround(cx)), static_cast<int>(std::lround(cy)),
              ax, ay, 255, 220, 0);
    draw_cross(rgb, cx, cy, 5, 0, 255, 255);
}
}  // namespace

int main(int argc, char** argv) {
    std::string data_dir = (argc > 1) ? argv[1] : "data/generated";
    std::string out_dir = (argc > 2) ? argv[2] : "results";
    fs::create_directories(out_dir);

    std::cout << "=========================================================\n";
    std::cout << "  Wafer AOI & Sub-Pixel Metrology C++17 Benchmark Suite  \n";
    std::cout << "=========================================================\n\n";

    std::ostringstream json;
    json << std::fixed << std::setprecision(4);
    json << "{\n";

    // =========================================================================
    // 1. Pyramid Shape-Based Template Matching (Single-Thread vs OpenMP)
    // =========================================================================
    std::cout << "[1/4] Pyramid Shape-Based Template Matching...\n";
    ImageU8 templ_u8 = load_pgm(data_dir + "/match_template.pgm");
    ImageU8 scene_u8 = load_pgm(data_dir + "/match_scene.pgm");
    ImageF32 templ_f = to_float(templ_u8);
    ImageF32 scene_f = to_float(scene_u8);

    ShapeMatchConfig mcfg;
    mcfg.num_levels = 3;
    mcfg.angle_min = -180.0f;
    mcfg.angle_max = 180.0f;
    mcfg.angle_step = 2.0f;
    mcfg.grad_thresh = 22.0f;
    mcfg.max_features_l0 = 360;
    mcfg.min_score = 0.62f;
    mcfg.greediness = 0.78f;
    mcfg.max_matches = 5;
    mcfg.nms_dist = 35.0f;

    // Single-threaded timing
    mcfg.use_openmp = false;
    ShapeBasedMatcher matcher_st;
    matcher_st.train(templ_f, mcfg);
    auto t0 = std::chrono::steady_clock::now();
    auto matches_st = matcher_st.match(scene_f);
    double match_st_ms = ms_since(t0);

    // Multi-threaded (OpenMP) timing
    mcfg.use_openmp = true;
    ShapeBasedMatcher matcher_mt;
    matcher_mt.train(templ_f, mcfg);
    t0 = std::chrono::steady_clock::now();
    auto matches_mt = matcher_mt.match(scene_f);
    double match_mt_ms = ms_since(t0);
    double speedup = match_st_ms / std::max(0.01, match_mt_ms);

    std::cout << "  Found " << matches_mt.size() << " IC dies | "
              << "Single-Thread: " << std::fixed << std::setprecision(2) << match_st_ms
              << " ms | OpenMP: " << match_mt_ms << " ms (" << speedup << "x speedup)\n";

    ImageU8 match_vis = gray_to_rgb(scene_u8);
    json << "  \"shape_matching\": {\n"
         << "    \"single_thread_ms\": " << match_st_ms << ",\n"
         << "    \"openmp_ms\": " << match_mt_ms << ",\n"
         << "    \"speedup\": " << speedup << ",\n"
         << "    \"matches\": [\n";
    for (size_t i = 0; i < matches_mt.size(); ++i) {
        const auto& m = matches_mt[i];
        std::cout << "    Die #" << (i + 1) << ": x=" << m.x << ", y=" << m.y
                  << ", angle=" << m.angle_deg << " deg, score=" << m.score << "\n";
        draw_rotated_box(match_vis, m.x, m.y,
                         static_cast<float>(templ_u8.width()),
                         static_cast<float>(templ_u8.height()),
                         m.angle_deg, 0, 255, 80);
        json << "      {\"x\": " << m.x << ", \"y\": " << m.y
             << ", \"angle_deg\": " << m.angle_deg << ", \"score\": " << m.score << "}"
             << (i + 1 < matches_mt.size() ? "," : "") << "\n";
    }
    json << "    ]\n  },\n";
    save_ppm(out_dir + "/01_shape_matching.ppm", match_vis);

    // =========================================================================
    // 2. Sub-Pixel Metrology & 30-Trial GR&R Repeatability Benchmark
    // =========================================================================
    std::cout << "\n[2/4] Sub-Pixel Edge Metrology & 30-Trial GR&R Repeatability...\n";
    ImageU8 bga_u8 = load_pgm(data_dir + "/metrology_bga.pgm");
    ImageF32 bga_f = to_float(bga_u8);
    t0 = std::chrono::steady_clock::now();
    auto bga_edges = extract_subpixel_edges(bga_f, 22.0f, 0.9f);
    auto bga_circle = fit_circle_ransac(bga_edges, 0.45f, 800, 42);
    double bga_ms = ms_since(t0);

    ImageU8 pins_u8 = load_pgm(data_dir + "/metrology_pins.pgm");
    ImageF32 pins_f = to_float(pins_u8);
    auto left_edges = extract_subpixel_edges(pins_f, 22.0f, 0.9f, 70, 35, 45, 180);
    auto right_edges = extract_subpixel_edges(pins_f, 22.0f, 0.9f, 130, 35, 45, 180);
    auto line_l = fit_line_ransac(left_edges, 0.45f, 600, 42);
    auto line_r = fit_line_ransac(right_edges, 0.45f, 600, 42);
    float measured_pitch = measure_parallel_line_pitch(line_l, line_r, 125.0f, 125.0f);

    // 30-trial repeatability across noisy frames (true radius = 42.35 px)
    std::vector<float> radius_trials, cx_trials, cy_trials;
    for (int k = 0; k < 30; ++k) {
        std::ostringstream fn;
        fn << data_dir << "/grr_bga_" << std::setw(2) << std::setfill('0') << k << ".pgm";
        if (!fs::exists(fn.str())) break;
        ImageF32 tf = to_float(load_pgm(fn.str()));
        auto pts = extract_subpixel_edges(tf, 22.0f, 0.9f);
        auto c = fit_circle_ransac(pts, 0.45f, 600, 100 + k);
        if (c.valid) {
            radius_trials.push_back(c.radius);
            cx_trials.push_back(c.cx);
            cy_trials.push_back(c.cy);
        }
    }
    auto rep_r = compute_repeatability(radius_trials, 42.35f);
    auto rep_x = compute_repeatability(cx_trials, 128.42f);

    std::cout << "  BGA Circle: cx=" << bga_circle.cx << ", cy=" << bga_circle.cy
              << ", r=" << bga_circle.radius << " px (RMS=" << bga_circle.rms_error
              << " px, time=" << bga_ms << " ms)\n";
    std::cout << "  IC Lead Width Pitch: " << measured_pitch << " px\n";
    std::cout << "  30-Trial GR&R Repeatability: Radius 3-sigma = " << rep_r.three_sigma
              << " px | Center-X 3-sigma = " << rep_x.three_sigma << " px\n";

    ImageU8 bga_vis = gray_to_rgb(bga_u8);
    for (const auto& p : bga_edges) {
        draw_pixel(bga_vis, static_cast<int>(std::lround(p.x)),
                   static_cast<int>(std::lround(p.y)), 0, 200, 255);
    }
    if (bga_circle.valid) {
        draw_circle(bga_vis, bga_circle.cx, bga_circle.cy, bga_circle.radius, 0, 255, 80);
        draw_cross(bga_vis, bga_circle.cx, bga_circle.cy, 6, 255, 60, 60);
    }
    save_ppm(out_dir + "/02_metrology_bga.ppm", bga_vis);

    json << "  \"metrology\": {\n"
         << "    \"bga_cx\": " << bga_circle.cx << ",\n"
         << "    \"bga_cy\": " << bga_circle.cy << ",\n"
         << "    \"bga_radius\": " << bga_circle.radius << ",\n"
         << "    \"bga_rms_px\": " << bga_circle.rms_error << ",\n"
         << "    \"bga_time_ms\": " << bga_ms << ",\n"
         << "    \"ic_lead_pitch_px\": " << measured_pitch << ",\n"
         << "    \"grr_trials\": " << radius_trials.size() << ",\n"
         << "    \"radius_mean_px\": " << rep_r.mean << ",\n"
         << "    \"radius_3sigma_px\": " << rep_r.three_sigma << ",\n"
         << "    \"center_x_3sigma_px\": " << rep_x.three_sigma << "\n"
         << "  },\n";

    // =========================================================================
    // 3. FFT Frequency-Domain Periodic Background Suppression
    // =========================================================================
    std::cout << "\n[3/4] FFT Frequency-Domain Periodic Wafer Background Suppression...\n";
    ImageU8 per_u8 = load_pgm(data_dir + "/wafer_periodic.pgm");
    ImageF32 per_f = to_float(per_u8);
    t0 = std::chrono::steady_clock::now();
    auto fft_res = fft_suppress_periodic_background(per_f, 10.0f, 3.6f, 4);
    double fft_ms = ms_since(t0);

    std::cout << "  Detected & notched " << fft_res.num_notches
              << " periodic spectral harmonics in " << fft_ms << " ms\n";
    save_pgm(out_dir + "/03_fft_spec_before.pgm", to_u8_normalized(fft_res.log_spectrum_before));
    save_pgm(out_dir + "/03_fft_spec_after.pgm", to_u8_normalized(fft_res.log_spectrum_after));
    save_pgm(out_dir + "/03_fft_filtered.pgm", to_u8_clamped(fft_res.filtered));

    json << "  \"fft_notch\": {\n"
         << "    \"num_notches\": " << fft_res.num_notches << ",\n"
         << "    \"time_ms\": " << fft_ms << "\n"
         << "  },\n";

    // =========================================================================
    // 4. Die-to-Die (D2D) Sub-Pixel Alignment & Defect Detection + Throughput
    // =========================================================================
    std::cout << "\n[4/4] Die-to-Die (D2D) Sub-Pixel Alignment & Defect Inspection...\n";
    ImageU8 gold_u8 = load_pgm(data_dir + "/d2d_golden.pgm");
    ImageU8 test_u8 = load_pgm(data_dir + "/d2d_test.pgm");
    ImageF32 gold_f = to_float(gold_u8);
    ImageF32 test_f = to_float(test_u8);

    // Ablation under high-sensitivity inspection threshold (base=11.0):
    // Naive fixed threshold (edge_tolerance_k = 0) vs Edge-Adaptive AOI (k = 0.22)
    D2DConfig naive_cfg;
    naive_cfg.base_diff_thresh = 11.0f;
    naive_cfg.edge_tolerance_k = 0.0f;
    naive_cfg.min_defect_area = 8;
    auto naive_res = inspect_die_to_die(gold_f, test_f, naive_cfg);

    D2DConfig aoi_cfg;
    aoi_cfg.base_diff_thresh = 11.0f;
    aoi_cfg.edge_tolerance_k = 0.22f;
    aoi_cfg.min_defect_area = 8;
    auto d2d_res = inspect_die_to_die(gold_f, test_f, aoi_cfg);

    std::cout << "  Sub-pixel D2D Alignment: dx=" << d2d_res.alignment.dx
              << " px, dy=" << d2d_res.alignment.dy << " px (time=" << d2d_res.elapsed_ms << " ms)\n";
    std::cout << "  Ablation -> Naive Diff detections: " << naive_res.defects.size()
              << " | Edge-Adaptive AOI detections: " << d2d_res.defects.size() << "\n";

    ImageU8 d2d_vis = gray_to_rgb(to_u8_clamped(d2d_res.aligned_test));
    for (const auto& d : d2d_res.defects) {
        std::cout << "    Defect #" << d.id << " [" << d.type << "] bbox=("
                  << d.x << "," << d.y << "," << d.w << "x" << d.h << ") area="
                  << d.area << " severity=" << d.max_severity << "\n";
        uint8_t r = 255, g = 50, b = 50;
        if (d.type == "Scratch") { r = 255; g = 180; b = 0; }
        else if (d.type == "Missing/LargeDefect") { r = 255; g = 40; b = 220; }
        draw_rect(d2d_vis, d.x - 3, d.y - 3, d.w + 6, d.h + 6, r, g, b, 2);
    }
    save_ppm(out_dir + "/04_d2d_defects.ppm", d2d_vis);
    save_pgm(out_dir + "/04_d2d_residual.pgm", to_u8_normalized(d2d_res.residual_map));
    save_pgm(out_dir + "/04_d2d_mask.pgm", d2d_res.defect_mask);

    // Multi-threaded production line throughput benchmark (16 frames via ThreadPool)
    constexpr int kBatchFrames = 16;
    ThreadPool pool(4);
    t0 = std::chrono::steady_clock::now();
    std::vector<std::future<size_t>> futs;
    futs.reserve(kBatchFrames);
    for (int i = 0; i < kBatchFrames; ++i) {
        futs.push_back(pool.enqueue([&]() {
            return inspect_die_to_die(gold_f, test_f, aoi_cfg).defects.size();
        }));
    }
    for (auto& f : futs) (void)f.get();
    double batch_ms = ms_since(t0);
    double fps = static_cast<double>(kBatchFrames) / (batch_ms * 1e-3);
    std::cout << "  4-Thread Pipeline Throughput (512x512 D2D): "
              << fps << " FPS (" << (batch_ms / kBatchFrames) << " ms/die)\n";

    json << "  \"d2d_inspection\": {\n"
         << "    \"est_dx\": " << d2d_res.alignment.dx << ",\n"
         << "    \"est_dy\": " << d2d_res.alignment.dy << ",\n"
         << "    \"single_die_ms\": " << d2d_res.elapsed_ms << ",\n"
         << "    \"naive_detections\": " << naive_res.defects.size() << ",\n"
         << "    \"adaptive_detections\": " << d2d_res.defects.size() << ",\n"
         << "    \"pipeline_4t_fps\": " << fps << ",\n"
         << "    \"pipeline_ms_per_die\": " << (batch_ms / kBatchFrames) << ",\n"
         << "    \"defects\": [\n";
    for (size_t i = 0; i < d2d_res.defects.size(); ++i) {
        const auto& d = d2d_res.defects[i];
        json << "      {\"id\": " << d.id << ", \"type\": \"" << d.type
             << "\", \"x\": " << d.x << ", \"y\": " << d.y
             << ", \"w\": " << d.w << ", \"h\": " << d.h
             << ", \"area\": " << d.area << ", \"severity\": " << d.max_severity << "}"
             << (i + 1 < d2d_res.defects.size() ? "," : "") << "\n";
    }
    json << "    ]\n  }\n}\n";

    std::ofstream jf(out_dir + "/benchmark_metrics.json");
    jf << json.str();
    std::cout << "\nSaved metrics to " << out_dir << "/benchmark_metrics.json\n";
    return 0;
}
