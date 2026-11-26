#!/usr/bin/env python3
"""
Generate realistic synthetic Semiconductor Wafer / IC AOI & Metrology datasets
with exact sub-pixel ground truth for C++17 benchmarking and unit validation.
"""
import json
import os
import math
import cv2
import numpy as np


def save_pgm(path: str, img_u8: np.ndarray) -> None:
    h, w = img_u8.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(img_u8.tobytes())


def generate_shape_matching_scene(out_dir: str, rng: np.random.Generator) -> dict:
    # 1. Construct realistic 96x96 IC QFN/Die template with asymmetric pin-1 fiducial
    S = 96
    templ = np.full((S, S), 38.0, dtype=np.float32)
    # Package body
    cv2.rectangle(templ, (16, 16), (S - 17, S - 17), 192.0, -1)
    # IC leads around perimeter
    for p in range(22, S - 20, 10):
        cv2.rectangle(templ, (p, 8), (p + 4, 15), 225.0, -1)
        cv2.rectangle(templ, (p, S - 16), (p + 4, S - 9), 225.0, -1)
        cv2.rectangle(templ, (8, p), (15, p + 4), 225.0, -1)
        cv2.rectangle(templ, (S - 16, p), (S - 9, p + 4), 225.0, -1)
    # Inner silicon die cavity
    cv2.rectangle(templ, (30, 30), (S - 31, S - 31), 68.0, -1)
    # Asymmetric pin-1 chamfer bar + fiducial dot (uniquely defines 0..360 orientation)
    cv2.rectangle(templ, (19, 19), (42, 30), 245.0, -1)
    cv2.circle(templ, (66, 26), 5, 25.0, -1)
    templ = cv2.GaussianBlur(templ, (3, 3), 0.7)
    save_pgm(os.path.join(out_dir, "match_template.pgm"), np.clip(templ, 0, 255).astype(np.uint8))

    # 2. Place 3 rotated & contrast-drifted IC dies onto a 640x480 inspection tray
    W, H = 640, 480
    scene = np.full((H, W), 38.0, dtype=np.float32)
    gt_poses = [
        {"x": 155.4, "y": 145.6, "angle_deg": -38.5},
        {"x": 452.7, "y": 158.3, "angle_deg": 27.2},
        {"x": 312.6, "y": 348.4, "angle_deg": 114.6},
    ]

    cx_t, cy_t = (S - 1) * 0.5, (S - 1) * 0.5
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    for pose in gt_poses:
        rad = math.radians(pose["angle_deg"])
        ca, sa = math.cos(rad), math.sin(rad)
        dx = xx - pose["x"]
        dy = yy - pose["y"]
        sx = ca * dx + sa * dy + cx_t
        sy = -sa * dx + ca * dy + cy_t
        mask = (sx >= 1) & (sx <= S - 2) & (sy >= 1) & (sy <= S - 2)
        warped = cv2.remap(templ, sx, sy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=38.0)
        scene[mask] = warped[mask]

    # Add realistic uneven bright-field illumination gradient + sensor noise
    illum = 0.72 + 0.35 * (xx / W) - 0.12 * (yy / H)
    scene = scene * illum + 14.0 + rng.normal(0.0, 3.2, size=(H, W)).astype(np.float32)
    save_pgm(os.path.join(out_dir, "match_scene.pgm"), np.clip(scene, 0, 255).astype(np.uint8))
    return {"poses": gt_poses}


def render_aa_circle(w: int, h: int, cx: float, cy: float, r: float, bg=35.0, fg=212.0) -> np.ndarray:
    S = 4
    yy, xx = np.mgrid[0 : h * S, 0 : w * S].astype(np.float32)
    px = (xx + 0.5) / S - 0.5
    py = (yy + 0.5) / S - 0.5
    dist = np.hypot(px - cx, py - cy)
    hi = np.where(dist <= r, fg, bg).astype(np.float32)
    return hi.reshape(h, S, w, S).mean(axis=(1, 3))


def render_aa_bar(w: int, h: int, x0: float, x1: float, y0: float, y1: float, bg=38.0, fg=215.0) -> np.ndarray:
    S = 4
    yy, xx = np.mgrid[0 : h * S, 0 : w * S].astype(np.float32)
    px = (xx + 0.5) / S - 0.5
    py = (yy + 0.5) / S - 0.5
    inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
    hi = np.where(inside, fg, bg).astype(np.float32)
    return hi.reshape(h, S, w, S).mean(axis=(1, 3))


def generate_metrology_dataset(out_dir: str, rng: np.random.Generator) -> dict:
    cx, cy, r = 128.42, 127.68, 42.35
    bga = render_aa_circle(256, 256, cx, cy, r)
    # Inject a realistic solder burr outlier on top boundary (tests RANSAC robustness!)
    bga_burr = bga.copy()
    cv2.circle(bga_burr, (128, 84), 6, 212.0, -1)
    bga_noisy = np.clip(bga_burr + rng.normal(0.0, 2.8, bga.shape), 0, 255).astype(np.uint8)
    save_pgm(os.path.join(out_dir, "metrology_bga.pgm"), bga_noisy)

    # 30 noisy frames for GR&R 3-sigma repeatability evaluation
    for k in range(30):
        frame = np.clip(bga_burr + rng.normal(0.0, 3.0, bga.shape), 0, 255).astype(np.uint8)
        save_pgm(os.path.join(out_dir, f"grr_bga_{k:02d}.pgm"), frame)

    # IC lead width / pitch measurement
    x_left = 92.35
    true_pitch = 56.72
    pins = render_aa_bar(256, 256, x_left, x_left + true_pitch, 24.0, 232.0)
    # Add small notch defect on one edge (tests RANSAC line outlier rejection)
    cv2.rectangle(pins, (89, 118), (97, 130), 38.0, -1)
    pins_noisy = np.clip(pins + rng.normal(0.0, 2.5, pins.shape), 0, 255).astype(np.uint8)
    save_pgm(os.path.join(out_dir, "metrology_pins.pgm"), pins_noisy)

    return {"bga_gt": {"cx": cx, "cy": cy, "radius": r}, "ic_lead_pitch_gt": true_pitch}


def generate_periodic_wafer(out_dir: str, rng: np.random.Generator) -> dict:
    N = 512
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    # Strong periodic memory-array grid (period = 16 px in X and Y + harmonic)
    bg = (
        125.0
        + 36.0 * np.sin(2.0 * math.pi * xx / 16.0)
        + 14.0 * np.sin(4.0 * math.pi * xx / 16.0)
        + 34.0 * np.sin(2.0 * math.pi * yy / 16.0)
        + 12.0 * np.sin(4.0 * math.pi * yy / 16.0)
    )
    # Inject faint non-periodic micro-scratch and particle hidden inside periodic pattern
    cv2.line(bg, (150, 180), (265, 240), 195.0, 2)
    cv2.circle(bg, (355, 310), 6, 225.0, -1)
    cv2.circle(bg, (210, 365), 5, 45.0, -1)
    bg = cv2.GaussianBlur(bg, (3, 3), 0.6)
    noisy = np.clip(bg + rng.normal(0.0, 2.0, (N, N)), 0, 255).astype(np.uint8)
    save_pgm(os.path.join(out_dir, "wafer_periodic.pgm"), noisy)
    return {"defects": ["micro_scratch_(150,180)-(265,240)", "bright_particle_(355,310)", "dark_pit_(210,365)"]}


def generate_d2d_pair(out_dir: str, rng: np.random.Generator) -> dict:
    N = 512
    golden = np.full((N, N), 58.0, dtype=np.float32)

    # Periodic wafer circuit grid + interconnects
    for x in range(32, N - 32, 32):
        cv2.rectangle(golden, (x, 28), (x + 6, N - 28), 155.0, -1)
    for y in range(32, N - 32, 32):
        cv2.rectangle(golden, (28, y), (N - 28, y + 6), 155.0, -1)

    # Bond pads around die perimeter
    pad_coords = []
    for k in range(56, N - 60, 56):
        pad_coords.append((k, 44, 24, 24))
        pad_coords.append((k, N - 68, 24, 24))
    for (px, py, pw, ph) in pad_coords:
        cv2.rectangle(golden, (px, py), (px + pw, py + ph), 210.0, -1)

    # Central logic macro block + alignment cross
    cv2.rectangle(golden, (168, 168), (344, 344), 108.0, -1)
    for y in range(182, 336, 18):
        cv2.rectangle(golden, (182, y), (330, y + 5), 185.0, -1)
    cv2.line(golden, (256, 224), (256, 288), 235.0, 4)
    cv2.line(golden, (224, 256), (288, 256), 235.0, 4)

    golden_clean = cv2.GaussianBlur(golden, (3, 3), 0.9)

    # Create test die with 4 realistic wafer defects BEFORE sub-pixel shift
    test = golden_clean.copy()
    gt_defects = []

    # Defect 1: Diagonal micro-scratch across memory interconnects
    cv2.line(test, (108, 118), (154, 148), 245.0, 2)
    gt_defects.append({"type": "Scratch", "cx": 131, "cy": 133})

    # Defect 2: Dark particle on bright circuit macro
    cv2.circle(test, (295, 220), 6, 22.0, -1)
    gt_defects.append({"type": "Particle", "cx": 295, "cy": 220})

    # Defect 3: Bright metallic particle in dark substrate
    cv2.circle(test, (392, 378), 5, 238.0, -1)
    gt_defects.append({"type": "Particle", "cx": 392, "cy": 378})

    # Defect 4: Missing bond pad (large defect)
    cv2.rectangle(test, (222, 42), (250, 70), 58.0, -1)
    gt_defects.append({"type": "Missing/LargeDefect", "cx": 236, "cy": 56})

    test = cv2.GaussianBlur(test, (3, 3), 0.6)

    # Apply exact sub-pixel translation (+5.38, -3.64) + 10% illumination gain/offset drift
    true_dx, true_dy = 5.38, -3.64
    M = np.array([[1.0, 0.0, true_dx], [0.0, 1.0, true_dy]], dtype=np.float32)
    test_shifted = cv2.warpAffine(test, M, (N, N), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    test_drifted = test_shifted * 1.10 + 12.0

    # Add realistic camera readout noise to both golden & test
    golden_u8 = np.clip(golden_clean + rng.normal(0.0, 1.8, (N, N)), 0, 255).astype(np.uint8)
    test_u8 = np.clip(test_drifted + rng.normal(0.0, 2.2, (N, N)), 0, 255).astype(np.uint8)

    save_pgm(os.path.join(out_dir, "d2d_golden.pgm"), golden_u8)
    save_pgm(os.path.join(out_dir, "d2d_test.pgm"), test_u8)

    return {
        "true_shift": {"dx": true_dx, "dy": true_dy},
        "illumination_drift": {"gain": 1.10, "offset": 12.0},
        "defects": gt_defects,
    }


def main():
    out_dir = os.path.join("data", "generated")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(2026)

    gt = {
        "shape_matching": generate_shape_matching_scene(out_dir, rng),
        "metrology": generate_metrology_dataset(out_dir, rng),
        "fft_periodic": generate_periodic_wafer(out_dir, rng),
        "d2d_inspection": generate_d2d_pair(out_dir, rng),
    }
    with open(os.path.join(out_dir, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    print(f"Generated synthetic semiconductor AOI dataset in {out_dir}")


if __name__ == "__main__":
    main()
