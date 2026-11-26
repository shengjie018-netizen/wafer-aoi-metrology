#include "wafer_aoi/shape_matcher.hpp"
#include "wafer_aoi/image_ops.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace wafer_aoi {

namespace {
constexpr float kPi = 3.14159265358979323846f;

void compute_unit_gradients(const ImageF32& img, float mag_min,
                            ImageF32& nx_map, ImageF32& ny_map) {
    ImageF32 smooth = gaussian_blur(img, 0.8f);
    ImageF32 gx, gy, mag;
    sobel_gradients(smooth, gx, gy, mag);
    const int w = img.width();
    const int h = img.height();
    nx_map = ImageF32(w, h, 1, 0.0f);
    ny_map = ImageF32(w, h, 1, 0.0f);
    for (size_t i = 0; i < mag.size(); ++i) {
        float m = mag.data()[i];
        if (m >= mag_min) {
            float inv = 1.0f / m;
            nx_map.data()[i] = gx.data()[i] * inv;
            ny_map.data()[i] = gy.data()[i] * inv;
        }
    }
}

float wrap_angle_deg(float a) {
    while (a > 180.0f) a -= 360.0f;
    while (a <= -180.0f) a += 360.0f;
    return a;
}
}  // namespace

std::vector<TemplateFeature> ShapeBasedMatcher::extract_features(
    const ImageF32& img, float grad_thresh, int max_features) {
    ImageF32 smooth = gaussian_blur(img, 0.8f);
    ImageF32 gx, gy, mag;
    sobel_gradients(smooth, gx, gy, mag);

    const int w = img.width();
    const int h = img.height();
    const float cx = (w - 1) * 0.5f;
    const float cy = (h - 1) * 0.5f;

    std::vector<TemplateFeature> candidates;
    candidates.reserve(static_cast<size_t>(w * h / 8));

    // Directional NMS along gradient normal to extract crisp 1-pixel contour ridges
    for (int y = 2; y < h - 2; ++y) {
        for (int x = 2; x < w - 2; ++x) {
            float m = mag.at(x, y);
            if (m < grad_thresh) continue;
            float nx = gx.at(x, y) / m;
            float ny = gy.at(x, y) / m;
            float m_pos = bilinear_sample(mag, x + nx, y + ny);
            float m_neg = bilinear_sample(mag, x - nx, y - ny);
            if (m >= m_pos && m >= m_neg && (m > m_pos || m > m_neg)) {
                candidates.push_back(TemplateFeature{
                    static_cast<float>(x) - cx,
                    static_cast<float>(y) - cy,
                    nx, ny});
            }
        }
    }

    // Uniform spatial subsampling if above feature cap
    std::vector<TemplateFeature> selected;
    if (static_cast<int>(candidates.size()) <= max_features) {
        selected = std::move(candidates);
    } else {
        selected.reserve(max_features);
        double step = static_cast<double>(candidates.size()) / static_cast<double>(max_features);
        for (int i = 0; i < max_features; ++i) {
            size_t idx = static_cast<size_t>(i * step);
            if (idx < candidates.size()) selected.push_back(candidates[idx]);
        }
    }

    // Deterministic shuffle so early-termination checks spatially distributed features first
    std::mt19937 rng(12345);
    std::shuffle(selected.begin(), selected.end(), rng);
    return selected;
}

std::vector<TemplateFeature> ShapeBasedMatcher::transform_features(
    const std::vector<TemplateFeature>& feats, float angle_deg, float scale) {
    const float rad = angle_deg * kPi / 180.0f;
    const float cos_a = std::cos(rad);
    const float sin_a = std::sin(rad);
    std::vector<TemplateFeature> out;
    out.reserve(feats.size());
    for (const auto& f : feats) {
        TemplateFeature tf;
        tf.dx = (f.dx * cos_a - f.dy * sin_a) * scale;
        tf.dy = (f.dx * sin_a + f.dy * cos_a) * scale;
        tf.nx = f.nx * cos_a - f.ny * sin_a;
        tf.ny = f.nx * sin_a + f.ny * cos_a;
        out.push_back(tf);
    }
    return out;
}

void ShapeBasedMatcher::train(const ImageF32& templ, const ShapeMatchConfig& cfg) {
    cfg_ = cfg;
    cfg_.num_levels = std::max(1, cfg_.num_levels);
    templ_w_ = templ.width();
    templ_h_ = templ.height();

    std::vector<ImageF32> pyr = build_pyramid(templ, cfg_.num_levels);
    base_features_.clear();
    base_features_.resize(cfg_.num_levels);

    for (int l = 0; l < cfg_.num_levels; ++l) {
        int cap = std::max(40, cfg_.max_features_l0 >> l);
        float thresh = std::max(10.0f, cfg_.grad_thresh * std::pow(0.85f, static_cast<float>(l)));
        base_features_[l] = extract_features(pyr[l], thresh, cap);
    }

    // Precompute coarse-level pose table
    coarse_poses_.clear();
    const int coarse_l = cfg_.num_levels - 1;
    float coarse_angle_step = cfg_.angle_step * static_cast<float>(1 << std::max(0, coarse_l - 1));
    coarse_angle_step = std::clamp(coarse_angle_step, cfg_.angle_step, 6.0f);

    std::vector<float> scales;
    if (cfg_.scale_max <= cfg_.scale_min + 1e-4f || cfg_.scale_step <= 1e-4f) {
        scales.push_back(cfg_.scale_min);
    } else {
        for (float s = cfg_.scale_min; s <= cfg_.scale_max + 1e-4f; s += cfg_.scale_step) {
            scales.push_back(s);
        }
    }

    for (float s : scales) {
        for (float a = cfg_.angle_min; a < cfg_.angle_max - 1e-4f; a += coarse_angle_step) {
            PrecomputedPoseTemplate pose;
            pose.angle_deg = a;
            pose.scale = s;
            pose.features = transform_features(base_features_[coarse_l], a, s);
            float max_x = 0.0f, max_y = 0.0f;
            for (const auto& f : pose.features) {
                max_x = std::max(max_x, std::abs(f.dx));
                max_y = std::max(max_y, std::abs(f.dy));
            }
            pose.half_w = max_x;
            pose.half_h = max_y;
            coarse_poses_.push_back(std::move(pose));
        }
        // Include angle_max if searching a narrow range (not full 360)
        if (cfg_.angle_max - cfg_.angle_min < 359.0f) {
            PrecomputedPoseTemplate pose;
            pose.angle_deg = cfg_.angle_max;
            pose.scale = s;
            pose.features = transform_features(base_features_[coarse_l], cfg_.angle_max, s);
            float max_x = 0.0f, max_y = 0.0f;
            for (const auto& f : pose.features) {
                max_x = std::max(max_x, std::abs(f.dx));
                max_y = std::max(max_y, std::abs(f.dy));
            }
            pose.half_w = max_x;
            pose.half_h = max_y;
            coarse_poses_.push_back(std::move(pose));
        }
    }
}

float ShapeBasedMatcher::score_pose(const ImageF32& nx_map, const ImageF32& ny_map,
                                    const std::vector<TemplateFeature>& feats,
                                    float cx, float cy, float min_accept_score,
                                    bool allow_early_exit) const {
    const int n = static_cast<int>(feats.size());
    if (n == 0) return 0.0f;
    const int w = nx_map.width();
    const int h = nx_map.height();
    const float* nx_data = nx_map.data();
    const float* ny_data = ny_map.data();

    float sum = 0.0f;
    const float min_total = min_accept_score * static_cast<float>(n);
    const float greedy_floor = min_accept_score * cfg_.greediness * 0.75f;

    for (int i = 0; i < n; ++i) {
        int sx = static_cast<int>(std::lround(cx + feats[i].dx));
        int sy = static_cast<int>(std::lround(cy + feats[i].dy));
        if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
            size_t idx = static_cast<size_t>(sy) * static_cast<size_t>(w) + static_cast<size_t>(sx);
            float dot = feats[i].nx * nx_data[idx] + feats[i].ny * ny_data[idx];
            if (cfg_.ignore_polarity) {
                sum += std::abs(dot);
            } else {
                sum += dot;
            }
        }
        // Check exact & greedy early exit every 12 features
        if (allow_early_exit && ((i & 15) == 15)) {
            int rem = n - 1 - i;
            if (sum + static_cast<float>(rem) < min_total) {
                return sum / static_cast<float>(n);
            }
            if (i >= 31 && sum < greedy_floor * static_cast<float>(i + 1)) {
                return sum / static_cast<float>(n);
            }
        }
    }
    return sum / static_cast<float>(n);
}

std::vector<ShapeMatchResult> ShapeBasedMatcher::match(const ImageF32& scene) const {
    if (scene.empty() || base_features_.empty() || coarse_poses_.empty()) {
        return {};
    }

    const int L = cfg_.num_levels - 1;
    std::vector<ImageF32> scene_pyr = build_pyramid(scene, cfg_.num_levels);
    std::vector<ImageF32> nx_pyr(cfg_.num_levels), ny_pyr(cfg_.num_levels);
    for (int l = 0; l < cfg_.num_levels; ++l) {
        compute_unit_gradients(scene_pyr[l], 5.0f, nx_pyr[l], ny_pyr[l]);
    }

    // 1. Coarse level exhaustive search (parallelized over pose table)
    const int cw = nx_pyr[L].width();
    const int ch = nx_pyr[L].height();
    const float coarse_thresh = std::max(0.30f, cfg_.min_score * 0.72f);

    std::vector<std::vector<ShapeMatchResult>> thread_candidates(coarse_poses_.size());

#ifdef WAFER_AOI_OPENMP
#pragma omp parallel for schedule(dynamic) if (cfg_.use_openmp && coarse_poses_.size() > 1)
#endif
    for (int p = 0; p < static_cast<int>(coarse_poses_.size()); ++p) {
        const auto& pose = coarse_poses_[p];
        int margin_x = std::max(2, static_cast<int>(std::floor(pose.half_w * 0.5f)));
        int margin_y = std::max(2, static_cast<int>(std::floor(pose.half_h * 0.5f)));
        auto& local_cands = thread_candidates[p];

        for (int y = margin_y; y < ch - margin_y; ++y) {
            for (int x = margin_x; x < cw - margin_x; ++x) {
                float sc = score_pose(nx_pyr[L], ny_pyr[L], pose.features,
                                      static_cast<float>(x), static_cast<float>(y),
                                      coarse_thresh, true);
                if (sc >= coarse_thresh) {
                    local_cands.push_back(ShapeMatchResult{
                        static_cast<float>(x), static_cast<float>(y),
                        pose.angle_deg, pose.scale, sc});
                }
            }
        }
    }

    std::vector<ShapeMatchResult> coarse_hits;
    for (auto& vec : thread_candidates) {
        coarse_hits.insert(coarse_hits.end(), vec.begin(), vec.end());
    }
    std::sort(coarse_hits.begin(), coarse_hits.end(),
              [](const ShapeMatchResult& a, const ShapeMatchResult& b) {
                  return a.score > b.score;
              });

    // Spatial NMS at coarse level to keep top distinct seeds
    std::vector<ShapeMatchResult> seeds;
    const float coarse_nms = std::max(4.0f, cfg_.nms_dist / static_cast<float>(1 << L));
    const float coarse_nms2 = coarse_nms * coarse_nms;
    const size_t max_seeds = static_cast<size_t>(std::max(cfg_.max_matches * 6, 30));

    for (const auto& cand : coarse_hits) {
        bool dup = false;
        for (const auto& s : seeds) {
            float dx = cand.x - s.x;
            float dy = cand.y - s.y;
            if (dx * dx + dy * dy < coarse_nms2 &&
                std::abs(wrap_angle_deg(cand.angle_deg - s.angle_deg)) < 20.0f) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            seeds.push_back(cand);
            if (seeds.size() >= max_seeds) break;
        }
    }

    // 2. Coarse-to-fine pyramid refinement down to Level 0
    std::vector<ShapeMatchResult> refined;
    refined.reserve(seeds.size());

    float init_angle_step = cfg_.angle_step * static_cast<float>(1 << std::max(0, L - 1));
    init_angle_step = std::clamp(init_angle_step, cfg_.angle_step, 6.0f);

    for (auto cand : seeds) {
        float cur_angle_step = init_angle_step;
        for (int l = L - 1; l >= 0; --l) {
            cand.x *= 2.0f;
            cand.y *= 2.0f;

            float fine_step = std::max(0.5f, cur_angle_step * 0.5f);
            float angle_radius = cur_angle_step;
            const int xy_rad = (l == 0) ? 2 : 3;

            ShapeMatchResult best = cand;
            best.score = -1.0f;

            for (float da = -angle_radius; da <= angle_radius + 1e-4f; da += fine_step) {
                float a = cand.angle_deg + da;
                auto feats = transform_features(base_features_[l], a, cand.scale);
                for (int dy = -xy_rad; dy <= xy_rad; ++dy) {
                    for (int dx = -xy_rad; dx <= xy_rad; ++dx) {
                        float tx = cand.x + static_cast<float>(dx);
                        float ty = cand.y + static_cast<float>(dy);
                        float sc = score_pose(nx_pyr[l], ny_pyr[l], feats, tx, ty,
                                              0.0f, false);
                        if (sc > best.score) {
                            best = ShapeMatchResult{tx, ty, a, cand.scale, sc};
                        }
                    }
                }
            }
            cand = best;
            cur_angle_step = fine_step;
        }

        // 3. Level-0 sub-pixel & sub-degree parabolic interpolation
        if (cand.score >= cfg_.min_score) {
            auto feats0 = transform_features(base_features_[0], cand.angle_deg, cand.scale);
            float sx_m = score_pose(nx_pyr[0], ny_pyr[0], feats0, cand.x - 1.0f, cand.y, 0.0f, false);
            float sx_0 = cand.score;
            float sx_p = score_pose(nx_pyr[0], ny_pyr[0], feats0, cand.x + 1.0f, cand.y, 0.0f, false);
            float denom_x = sx_m - 2.0f * sx_0 + sx_p;
            if (denom_x < -1e-6f) {
                cand.x += std::clamp(0.5f * (sx_m - sx_p) / denom_x, -0.5f, 0.5f);
            }

            float sy_m = score_pose(nx_pyr[0], ny_pyr[0], feats0, cand.x, cand.y - 1.0f, 0.0f, false);
            float sy_p = score_pose(nx_pyr[0], ny_pyr[0], feats0, cand.x, cand.y + 1.0f, 0.0f, false);
            float denom_y = sy_m - 2.0f * sx_0 + sy_p;
            if (denom_y < -1e-6f) {
                cand.y += std::clamp(0.5f * (sy_m - sy_p) / denom_y, -0.5f, 0.5f);
            }

            const float a_step = 0.5f;
            auto feats_am = transform_features(base_features_[0], cand.angle_deg - a_step, cand.scale);
            auto feats_ap = transform_features(base_features_[0], cand.angle_deg + a_step, cand.scale);
            float sa_m = score_pose(nx_pyr[0], ny_pyr[0], feats_am, cand.x, cand.y, 0.0f, false);
            float sa_p = score_pose(nx_pyr[0], ny_pyr[0], feats_ap, cand.x, cand.y, 0.0f, false);
            float denom_a = sa_m - 2.0f * sx_0 + sa_p;
            if (denom_a < -1e-6f) {
                cand.angle_deg += a_step * std::clamp(0.5f * (sa_m - sa_p) / denom_a, -0.5f, 0.5f);
            }
            cand.angle_deg = wrap_angle_deg(cand.angle_deg);
            refined.push_back(cand);
        }
    }

    // Final 2D NMS
    std::sort(refined.begin(), refined.end(),
              [](const ShapeMatchResult& a, const ShapeMatchResult& b) {
                  return a.score > b.score;
              });

    std::vector<ShapeMatchResult> results;
    const float nms2 = cfg_.nms_dist * cfg_.nms_dist;
    for (const auto& r : refined) {
        bool dup = false;
        for (const auto& kept : results) {
            float dx = r.x - kept.x;
            float dy = r.y - kept.y;
            if (dx * dx + dy * dy < nms2) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            results.push_back(r);
            if (static_cast<int>(results.size()) >= cfg_.max_matches) break;
        }
    }
    return results;
}

}  // namespace wafer_aoi
