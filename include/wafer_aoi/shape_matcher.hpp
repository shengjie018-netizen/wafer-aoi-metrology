#pragma once

#include <vector>
#include "wafer_aoi/image.hpp"

namespace wafer_aoi {

struct TemplateFeature {
    float dx = 0.0f;  // offset from template center
    float dy = 0.0f;
    float nx = 0.0f;  // unit gradient x
    float ny = 0.0f;  // unit gradient y
};

struct PrecomputedPoseTemplate {
    float angle_deg = 0.0f;
    float scale = 1.0f;
    std::vector<TemplateFeature> features;
    float half_w = 0.0f;
    float half_h = 0.0f;
};

struct ShapeMatchConfig {
    int num_levels = 3;          // pyramid levels (0 = finest, num_levels-1 = coarsest)
    float angle_min = -180.0f;   // search angle range in degrees
    float angle_max = 180.0f;
    float angle_step = 2.0f;     // coarse angle step in degrees
    float scale_min = 1.0f;      // scale range
    float scale_max = 1.0f;
    float scale_step = 0.05f;
    float grad_thresh = 25.0f;   // template edge gradient threshold
    int max_features_l0 = 400;   // feature cap at finest level for speed
    float min_score = 0.65f;     // acceptance threshold in [0, 1]
    float greediness = 0.75f;    // Halcon-style early-exit greediness [0, 1]
    int max_matches = 10;
    float nms_dist = 20.0f;      // non-maximum suppression radius (px)
    bool use_openmp = true;      // toggle OpenMP for single vs multi-thread benchmark
    bool ignore_polarity = false;  // true = use |dot| (contrast reversal invariant)
};

struct ShapeMatchResult {
    float x = 0.0f;          // matched template center x (sub-pixel)
    float y = 0.0f;          // matched template center y (sub-pixel)
    float angle_deg = 0.0f;  // matched rotation angle (sub-degree)
    float scale = 1.0f;
    float score = 0.0f;
};

class ShapeBasedMatcher {
public:
    // Train pyramid shape model from grayscale template image
    void train(const ImageF32& templ, const ShapeMatchConfig& cfg);

    // Run coarse-to-fine pyramid matching with sub-pixel/sub-degree refinement
    std::vector<ShapeMatchResult> match(const ImageF32& scene) const;

    // Convenience stats for inspection/benchmarking
    int template_width() const { return templ_w_; }
    int template_height() const { return templ_h_; }
    size_t num_features(int level) const {
        return (level >= 0 && level < static_cast<int>(base_features_.size()))
                   ? base_features_[level].size()
                   : 0;
    }

private:
    ShapeMatchConfig cfg_;
    int templ_w_ = 0;
    int templ_h_ = 0;

    // Base unrotated features at each pyramid level
    std::vector<std::vector<TemplateFeature>> base_features_;
    // Precomputed rotated/scaled pose table at coarsest level
    std::vector<PrecomputedPoseTemplate> coarse_poses_;

    static std::vector<TemplateFeature> extract_features(
        const ImageF32& img, float grad_thresh, int max_features);

    static std::vector<TemplateFeature> transform_features(
        const std::vector<TemplateFeature>& feats, float angle_deg, float scale);

    float score_pose(const ImageF32& nx_map, const ImageF32& ny_map,
                     const std::vector<TemplateFeature>& feats,
                     float cx, float cy, float min_accept_score,
                     bool allow_early_exit) const;
};

}  // namespace wafer_aoi
