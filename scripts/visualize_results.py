#!/usr/bin/env python3
"""
Render publication-grade visual panels (PNG) for GitHub README from C++ AOI outputs.
Pure OpenCV + NumPy (zero matplotlib dependency).
"""
import json
import os
import cv2
import numpy as np


def add_banner(img_bgr: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bar_h = 54 if subtitle else 36
    canvas = np.full((h + bar_h, w, 3), (24, 26, 32), dtype=np.uint8)
    canvas[bar_h:, :] = img_bgr
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 245, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 215, 255), 1, cv2.LINE_AA)
    return canvas


def hstack_panels(panels, pad=10):
    max_h = max(p.shape[0] for p in panels)
    out = []
    for i, p in enumerate(panels):
        if p.shape[0] < max_h:
            padded = np.full((max_h, p.shape[1], 3), (24, 26, 32), dtype=np.uint8)
            y0 = (max_h - p.shape[0]) // 2
            padded[y0 : y0 + p.shape[0], :] = p
            p = padded
        out.append(p)
        if i + 1 < len(panels):
            out.append(np.full((max_h, pad, 3), (24, 26, 32), dtype=np.uint8))
    return np.hstack(out)


def main():
    data_dir = os.path.join("data", "generated")
    res_dir = "results"
    asset_dir = os.path.join("docs", "assets")
    os.makedirs(asset_dir, exist_ok=True)

    with open(os.path.join(res_dir, "benchmark_metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Shape Matching panel
    templ = cv2.cvtColor(cv2.imread(os.path.join(data_dir, "match_template.pgm"), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    templ_big = cv2.resize(templ, (220, 220), interpolation=cv2.INTER_NEAREST)
    match_vis = cv2.imread(os.path.join(res_dir, "01_shape_matching.ppm"))
    sm = metrics["shape_matching"]
    for i, m in enumerate(sm["matches"]):
        txt = f"#{i+1} {m['angle_deg']:.1f}deg (s={m['score']:.2f})"
        cv2.putText(match_vis, txt, (int(m["x"]) - 65, int(m["y"]) - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 120), 1, cv2.LINE_AA)
    p1 = add_banner(templ_big, "Template (96x96)", "Gradient Contour Model")
    p2 = add_banner(match_vis, "Pyramid Shape-Based Matching (C++17 + OpenMP)",
                    f"1T: {sm['single_thread_ms']:.1f} ms | OpenMP: {sm['openmp_ms']:.1f} ms ({sm['speedup']:.2f}x speedup)")
    cv2.imwrite(os.path.join(asset_dir, "01_shape_matching.png"), hstack_panels([p1, p2]))

    # 2. Sub-Pixel Metrology panel
    bga_vis = cv2.imread(os.path.join(res_dir, "02_metrology_bga.ppm"))
    bga_big = cv2.resize(bga_vis, (360, 360), interpolation=cv2.INTER_NEAREST)
    # 4x zoom crop around top burr + sub-pixel circle edge
    zoom = bga_vis[65:135, 95:165]
    zoom_big = cv2.resize(zoom, (360, 360), interpolation=cv2.INTER_NEAREST)
    mt = metrics["metrology"]
    p1 = add_banner(bga_big, "RANSAC + Gauss-Newton Circle Fit",
                    f"r={mt['bga_radius']:.3f}px (GT 42.350) | 3-sigma={mt['radius_3sigma_px']:.4f}px")
    p2 = add_banner(zoom_big, "5x Zoom: Burr Outlier Rejection",
                    f"RMS={mt['bga_rms_px']:.4f}px | IC Pitch={mt['ic_lead_pitch_px']:.3f}px (GT 56.720)")
    cv2.imwrite(os.path.join(asset_dir, "02_subpixel_metrology.png"), hstack_panels([p1, p2]))

    # 3. FFT Periodic Suppression panel
    per_raw = cv2.cvtColor(cv2.imread(os.path.join(data_dir, "wafer_periodic.pgm"), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    spec_b = cv2.applyColorMap(cv2.imread(os.path.join(res_dir, "03_fft_spec_before.pgm"), cv2.IMREAD_GRAYSCALE), cv2.COLORMAP_INFERNO)
    spec_a = cv2.applyColorMap(cv2.imread(os.path.join(res_dir, "03_fft_spec_after.pgm"), cv2.IMREAD_GRAYSCALE), cv2.COLORMAP_INFERNO)
    per_flt = cv2.cvtColor(cv2.imread(os.path.join(res_dir, "03_fft_filtered.pgm"), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    sz = (300, 300)
    ft = metrics["fft_notch"]
    panels_fft = [
        add_banner(cv2.resize(per_raw, sz), "1. Raw Periodic Wafer", "Micro-defects hidden in stripes"),
        add_banner(cv2.resize(spec_b, sz), "2. 2D FFT Log Spectrum", "Bright harmonic spikes"),
        add_banner(cv2.resize(spec_a, sz), "3. Auto Notch Filter", f"{ft['num_notches']} harmonics suppressed"),
        add_banner(cv2.resize(per_flt, sz), "4. Filtered Output", f"Defects isolated ({ft['time_ms']:.1f} ms)"),
    ]
    cv2.imwrite(os.path.join(asset_dir, "03_fft_periodic_suppression.png"), hstack_panels(panels_fft))

    # 4. Die-to-Die Inspection panel
    gold = cv2.cvtColor(cv2.imread(os.path.join(data_dir, "d2d_golden.pgm"), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    res_map = cv2.applyColorMap(cv2.imread(os.path.join(res_dir, "04_d2d_residual.pgm"), cv2.IMREAD_GRAYSCALE), cv2.COLORMAP_JET)
    d2d_vis = cv2.imread(os.path.join(res_dir, "04_d2d_defects.ppm"))
    d2 = metrics["d2d_inspection"]
    for d in d2["defects"]:
        cv2.putText(d2d_vis, f"{d['type']} ({d['area']}px)", (max(4, d["x"] - 10), max(16, d["y"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 255), 1, cv2.LINE_AA)
    sz3 = (360, 360)
    panels_d2d = [
        add_banner(cv2.resize(gold, sz3), "1. Golden Reference Die", "512x512 Wafer Circuit"),
        add_banner(cv2.resize(res_map, sz3), "2. Sub-Pixel Aligned Residual",
                   f"Shift=({d2['est_dx']:.2f}, {d2['est_dy']:.2f})px (GT 5.38,-3.64)"),
        add_banner(cv2.resize(d2d_vis, sz3), "3. Edge-Adaptive D2D Detections",
                   f"Found {d2['adaptive_detections']}/4 (Naive: {d2['naive_detections']}) | {d2['pipeline_4t_fps']:.1f} FPS"),
    ]
    cv2.imwrite(os.path.join(asset_dir, "04_d2d_defect_inspection.png"), hstack_panels(panels_d2d))
    print(f"Rendered 4 README visual panels into {asset_dir}")


if __name__ == "__main__":
    main()
