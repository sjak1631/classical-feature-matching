#!/usr/bin/env python3
"""Brightness (contrast) change experiment for the SIFT scratch implementation.

Setup
-----
- img1 : the 500×500 source image (alpha = 1.0)
- img2 : img1 scaled by *alpha* (pixel values multiplied by alpha, clipped to
         [0, 255]).  alpha < 1 → darker;  alpha > 1 → brighter / saturated.

Evaluation region
-----------------
Brightness scaling introduces no geometric transformation, so the full
500×500 image is used as the evaluation region — every keypoint is counted.

Outputs
-------
  experiments/results/brightness/
    {stem}_alpha{alpha_str}.png   per-alpha visualisation
    summary.png                   inlier-rate vs alpha line chart
    stats.json                    full statistics (JSON)

Usage
-----
    uv run python experiments/brightness_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sift import (
    build_gaussian_pyramid,
    build_dog_pyramid,
    detect_keypoints,
    localize_keypoints,
    assign_orientations,
    compute_descriptors,
    match_descriptors,
    geometric_verification,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALPHAS: list[float] = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

IMAGE_NAMES = ["source1.jpg"]
IMAGES_DIR  = Path(__file__).parent / "inputs"
OUT_DIR     = Path(__file__).parent / "outputs" / "brightness"

SIFT_OCTAVES  = 4
SIFT_SCALES   = 3
SIFT_SIGMA    = 1.6
RATIO         = 0.75
AFFINE_THRESH = 6.0


def _alpha_str(alpha: float) -> str:
    """e.g. 0.4 → '040', 1.0 → '100', 2.0 → '200'"""
    return f"{int(round(alpha * 100)):03d}"


# ---------------------------------------------------------------------------
# Brightness adjustment
# ---------------------------------------------------------------------------

def adjust_brightness(bgr: np.ndarray, alpha: float) -> np.ndarray:
    """Return *bgr* with each channel scaled by *alpha*, clipped to [0, 255]."""
    return np.clip(bgr.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# SIFT pipeline
# ---------------------------------------------------------------------------

def run_sift(gray: np.ndarray) -> tuple[list[dict], np.ndarray]:
    gp    = build_gaussian_pyramid(gray, num_octaves=SIFT_OCTAVES,
                                   num_scales=SIFT_SCALES, sigma=SIFT_SIGMA)
    dp    = build_dog_pyramid(gp)
    cands = detect_keypoints(dp)
    kps   = localize_keypoints(dp, cands)
    kps   = assign_orientations(gp, kps)
    descs = compute_descriptors(gp, kps)
    return kps, descs


def kp_xy(kp: dict) -> tuple[float, float]:
    s = 2.0 ** kp["octave"]
    return kp["col"] * s / 2.0, kp["row"] * s / 2.0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_match_figure(
    bgr1: np.ndarray,
    bgr2: np.ndarray,
    kps1: list[dict],
    kps2: list[dict],
    matches: list[tuple[int, int, float]],
    inlier_mask: np.ndarray,
    stats: dict,
    title: str,
    out_path: Path,
) -> None:
    img1_vis = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
    img2_vis = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

    h1, w1 = img1_vis.shape[:2]
    h2, w2 = img2_vis.shape[:2]
    canvas  = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_vis
    canvas[:h2, w1:] = img2_vis

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(canvas)

    for i, (i1, i2, _) in enumerate(matches):
        x1, y1 = kp_xy(kps1[i1])
        x2, y2 = kp_xy(kps2[i2])
        x2 += w1
        is_inlier = bool(inlier_mask[i])
        color = "lime" if is_inlier else "red"
        alpha_v = 0.85 if is_inlier else 0.35
        lw      = 0.8  if is_inlier else 0.4
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha_v)
        ax.plot(x1, y1, ".", color=color, ms=3, alpha=alpha_v)
        ax.plot(x2, y2, ".", color=color, ms=3, alpha=alpha_v)

    # Labels under each image
    ax.text(w1 / 2, h1 + 8, "img1  (α = 1.0)", ha="center", va="top",
            fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.5, pad=2))
    ax.text(w1 + w2 / 2, h2 + 8, f"img2  (α = {stats['alpha']})",
            ha="center", va="top", fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.5, pad=2))

    ax.axis("off")
    ax.set_title(title, fontsize=9)

    n_out = stats["n_matches"] - stats["n_inliers"]
    legend = [
        mpatches.Patch(color="lime", label=f"Inlier  ({stats['n_inliers']})"),
        mpatches.Patch(color="red",  label=f"Outlier ({n_out})"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_summary_figure(all_stats: dict, out_path: Path) -> None:
    stems   = list(all_stats.keys())
    n_stems = len(stems)
    fig, axes = plt.subplots(1, n_stems, figsize=(8 * n_stems, 5))
    if n_stems == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")

    for ax, stem in zip(axes, stems):
        ds      = all_stats[stem]
        alphas  = ALPHAS
        rates   = [ds[_alpha_str(a)]["inlier_rate"] for a in alphas]
        inliers = [ds[_alpha_str(a)]["n_inliers"]   for a in alphas]
        matches = [ds[_alpha_str(a)]["n_matches"]    for a in alphas]
        kp2_cts = [ds[_alpha_str(a)]["n_kp2"]        for a in alphas]

        xs = range(len(alphas))
        
        # Left axis: inlier rate
        ax.plot(xs, rates, marker="o", color=cmap(0), lw=1.8, ms=6, label="Inlier rate")
        ax.set_ylabel("Inlier rate", color=cmap(0), fontsize=10)
        ax.tick_params(axis="y", labelcolor=cmap(0))
        ax.set_ylim(0, 1.20)
        
        # Right axis: match count
        ax2 = ax.twinx()
        ax2.plot(xs, matches, marker="s", color=cmap(1), lw=1.8, ms=5, label="Match count")
        ax2.set_ylabel("Match count", color=cmap(1), fontsize=10)
        ax2.tick_params(axis="y", labelcolor=cmap(1))
        ax2.set_ylim(0, max(matches) * 1.1 if matches else 1)
        
        # Far right axis: KP2 count
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.plot(xs, kp2_cts, marker="^", color=cmap(2), lw=1.8, ms=5, label="KP2 count")
        ax3.set_ylabel("KP2 count", color=cmap(2), fontsize=10)
        ax3.tick_params(axis="y", labelcolor=cmap(2))
        ax3.set_ylim(0, max(kp2_cts) * 1.1 if kp2_cts else 1)

        for i, (r, ni, nm) in enumerate(zip(rates, inliers, matches)):
            ax.text(i, r + 0.018, f"{ni}/{nm}",
                    ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(list(xs))
        ax.set_xticklabels([str(a) for a in alphas])
        ax.set_xlabel("Brightness scale α")
        ax.set_title(stem)
        ax.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.4)
        
        # Combine legends from all axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, fontsize=8, loc="lower right")

    fig.suptitle(
        "SIFT brightness invariance",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary → {out_path}")


# ---------------------------------------------------------------------------
# Per-image experiment
# ---------------------------------------------------------------------------

def run_one_image(bgr: np.ndarray, stem: str) -> dict:
    alpha_stats: dict = {}

    # img1 SIFT computed once
    gray1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    kps1, descs1 = run_sift(gray1)

    for alpha in ALPHAS:
        key = _alpha_str(alpha)
        print(f"  α={alpha:.2f}", end="  ", flush=True)
        t0 = time.perf_counter()

        bgr2  = adjust_brightness(bgr, alpha)
        gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

        if alpha == 1.0:
            kps2, descs2 = kps1, descs1
        else:
            kps2, descs2 = run_sift(gray2)

        matches = match_descriptors(descs1, descs2, ratio_threshold=RATIO)

        if len(matches) >= 4:
            _, inlier_mask = geometric_verification(
                kps1, kps2, matches, gray1.shape,
                affine_thresh=AFFINE_THRESH,
            )
        else:
            inlier_mask = np.zeros(len(matches), dtype=bool)

        n_m  = len(matches)
        n_in = int(inlier_mask.sum())
        rate = n_in / max(n_m, 1)

        elapsed = time.perf_counter() - t0
        print(
            f"kp={len(kps1)}+{len(kps2)}"
            f"  matches={n_m}  inliers={n_in}"
            f"  rate={rate:.1%}  [{elapsed:.1f}s]"
        )

        alpha_stats[key] = {
            "alpha":       alpha,
            "n_kp1":       len(kps1),
            "n_kp2":       len(kps2),
            "n_matches":   n_m,
            "n_inliers":   n_in,
            "inlier_rate": round(rate, 4),
            "elapsed_sec": round(elapsed, 2),
        }

        title = (
            f"{stem}  |  α = {alpha}  |  "
            f"kp: {len(kps1)}+{len(kps2)}  |  "
            f"matches: {n_m}  |  inliers: {n_in}  |  rate: {rate:.1%}"
        )
        save_match_figure(
            bgr, bgr2,
            kps1, kps2, matches, inlier_mask,
            alpha_stats[key], title,
            OUT_DIR / f"{stem}_alpha{key}.png",
        )

    return alpha_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats: dict = {}

    for img_name in IMAGE_NAMES:
        img_path = IMAGES_DIR / img_name
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"WARNING: cannot read {img_path}, skipping.")
            continue
        stem = Path(img_name).stem
        print(f"\n=== {stem}  ({bgr.shape[1]}×{bgr.shape[0]}) ===")
        all_stats[stem] = run_one_image(bgr, stem)

    stats_path = OUT_DIR / "brightness_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats  → {stats_path}")

    save_summary_figure(all_stats, OUT_DIR / "brightness_summary.png")


if __name__ == "__main__":
    main()
