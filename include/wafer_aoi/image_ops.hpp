#pragma once

#include <vector>
#include "wafer_aoi/image.hpp"

namespace wafer_aoi {

struct ConnectedComponent {
    int id = 0;
    int area = 0;
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    float cx = 0.0f;
    float cy = 0.0f;
    float aspect_ratio = 1.0f;  // max(w, h) / min(w, h)
    float fill_ratio = 0.0f;    // area / (w * h)
};

// Separable 2D Gaussian blur: O(2*k) per pixel
ImageF32 gaussian_blur(const ImageF32& src, float sigma);

// 2x Gaussian pyramid downsampling
ImageF32 pyr_down(const ImageF32& src);

// Build multi-level Gaussian pyramid (level 0 is original)
std::vector<ImageF32> build_pyramid(const ImageF32& src, int num_levels);

// 3x3 Sobel gradients: gx, gy, magnitude
void sobel_gradients(const ImageF32& src, ImageF32& gx, ImageF32& gy, ImageF32& mag);

// Bilinear interpolation sample with boundary clamping
float bilinear_sample(const ImageF32& img, float x, float y);

// Sub-pixel translation shift: dst(x, y) = src(x - dx, y - dy)
ImageF32 shift_subpixel(const ImageF32& src, float dx, float dy);

// Affine warp around center (cx, cy) with rotation (deg) and scale
ImageF32 warp_similarity(const ImageF32& src, int dst_w, int dst_h,
                         float tx, float ty, float angle_deg, float scale,
                         float cx, float cy);

// Integral image (width+1) x (height+1) of double for O(1) box sums
std::vector<double> compute_integral_image(const ImageF32& src);

// Otsu automatic global thresholding
uint8_t otsu_threshold(const ImageU8& src);

// Binary morphology (square kernel of size 2*radius + 1)
ImageU8 morph_erode(const ImageU8& binary, int radius);
ImageU8 morph_dilate(const ImageU8& binary, int radius);
ImageU8 morph_open(const ImageU8& binary, int radius);
ImageU8 morph_close(const ImageU8& binary, int radius);

// Grayscale morphological top-hat (src - open(src)) for bright spot enhancement
ImageF32 morph_tophat_gray(const ImageF32& src, int radius);

// 8-connectivity Two-Pass Union-Find connected component labeling
std::vector<ConnectedComponent> connected_components(
    const ImageU8& binary, int min_area, ImageI32* labels_out = nullptr);

}  // namespace wafer_aoi
