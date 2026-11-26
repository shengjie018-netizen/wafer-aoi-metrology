#pragma once

#include <complex>
#include <string>
#include <vector>
#include "wafer_aoi/image.hpp"
#include "wafer_aoi/image_ops.hpp"

namespace wafer_aoi {

using ComplexD = std::complex<double>;

struct PhaseShiftResult {
    float dx = 0.0f;         // shift of test relative to ref (test shifted by dx, dy)
    float dy = 0.0f;
    float peak_value = 0.0f; // correlation peak sharpness
};

struct D2DConfig {
    float base_diff_thresh = 18.0f;   // baseline grayscale residual threshold
    float edge_tolerance_k = 0.18f;   // gradient-adaptive tolerance to suppress edge overkill
    int border_ignore = 8;            // ignore image border pixels after sub-pixel shift
    int morph_open_radius = 1;        // remove single-pixel shot noise
    int morph_close_radius = 1;       // merge fragmented scratch pieces
    int min_defect_area = 6;          // minimum pixel area to report as defect
    bool normalize_illumination = true;
};

struct DefectInfo {
    int id = 0;
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    int area = 0;
    float cx = 0.0f;
    float cy = 0.0f;
    float max_severity = 0.0f;
    float aspect_ratio = 1.0f;
    std::string type;  // "Scratch", "Particle", "Missing/LargeDefect"
};

struct D2DInspectionResult {
    PhaseShiftResult alignment;
    ImageF32 aligned_test;
    ImageF32 residual_map;
    ImageU8 defect_mask;
    std::vector<DefectInfo> defects;
    double elapsed_ms = 0.0;
};

struct FFTNotchResult {
    ImageF32 filtered;
    ImageF32 log_spectrum_before;
    ImageF32 log_spectrum_after;
    int num_notches = 0;
};

// Pure C++17 Radix-2 Cooley-Tukey 1D & 2D FFT
void fft_1d(std::vector<ComplexD>& data, bool inverse);
void fft_2d(std::vector<ComplexD>& data, int width, int height, bool inverse);

// Sub-pixel Die-to-Die translation estimation via Hann-windowed Phase Correlation
PhaseShiftResult phase_correlate_subpixel(const ImageF32& ref, const ImageF32& test);

// FFT Frequency-Domain Notch Filter to suppress periodic wafer circuit background
FFTNotchResult fft_suppress_periodic_background(
    const ImageF32& src,
    float dc_keep_radius = 10.0f,
    float spike_zscore = 3.8f,
    int notch_radius = 4);

// Full Die-to-Die (D2D) Alignment + Illumination Norm + Edge-Adaptive Residual Defect Detection
D2DInspectionResult inspect_die_to_die(
    const ImageF32& golden,
    const ImageF32& test,
    const D2DConfig& cfg = D2DConfig{});

}  // namespace wafer_aoi
