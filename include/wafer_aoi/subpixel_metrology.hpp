#pragma once

#include <vector>
#include "wafer_aoi/image.hpp"

namespace wafer_aoi {

struct SubpixelPoint {
    float x = 0.0f;
    float y = 0.0f;
    float nx = 0.0f;  // unit gradient normal
    float ny = 0.0f;
    float mag = 0.0f;
};

// Normalized line: a*x + b*y + c = 0, with a^2 + b^2 = 1
struct Line2D {
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float rms_error = 0.0f;
    int inliers = 0;
    bool valid = false;

    float signed_distance(float x, float y) const { return a * x + b * y + c; }
    float distance(float x, float y) const {
        float d = signed_distance(x, y);
        return d >= 0.0f ? d : -d;
    }
};

struct Circle2D {
    float cx = 0.0f;
    float cy = 0.0f;
    float radius = 0.0f;
    float rms_error = 0.0f;
    int inliers = 0;
    bool valid = false;
};

struct RepeatabilityStats {
    float mean = 0.0f;
    float std_dev = 0.0f;
    float three_sigma = 0.0f;
    float max_abs_err = 0.0f;
};

// Sub-pixel edge extraction using directional NMS + 3-point quadratic interpolation
// along the continuous gradient normal.
std::vector<SubpixelPoint> extract_subpixel_edges(
    const ImageF32& img,
    float mag_thresh = 20.0f,
    float smooth_sigma = 1.0f,
    int roi_x = 0, int roi_y = 0, int roi_w = -1, int roi_h = -1);

// Total Least Squares (PCA) line fit
Line2D fit_line_tls(const std::vector<SubpixelPoint>& pts);

// RANSAC + TLS refit line estimation (robust to burrs/defects/outliers)
Line2D fit_line_ransac(const std::vector<SubpixelPoint>& pts,
                       float inlier_thresh = 0.5f,
                       int max_iters = 500,
                       uint32_t seed = 42);

// Kasa algebraic init + Gauss-Newton geometric circle refinement
Circle2D fit_circle_geometric(const std::vector<SubpixelPoint>& pts, int gn_iters = 15);

// RANSAC + Geometric Gauss-Newton circle estimation (e.g., BGA ball / wafer fiducial)
Circle2D fit_circle_ransac(const std::vector<SubpixelPoint>& pts,
                           float inlier_thresh = 0.5f,
                           int max_iters = 800,
                           uint32_t seed = 42);

// Measure perpendicular pitch between two approximately parallel lines at reference point
float measure_parallel_line_pitch(const Line2D& l1, const Line2D& l2,
                                  float ref_x, float ref_y);

// Measure Euclidean center-to-center distance between two circles
float measure_circle_center_dist(const Circle2D& c1, const Circle2D& c2);

// Compute 3-sigma repeatability stats against ground truth
RepeatabilityStats compute_repeatability(const std::vector<float>& measurements,
                                         float ground_truth);

}  // namespace wafer_aoi
