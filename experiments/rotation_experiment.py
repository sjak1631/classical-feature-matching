#!/usr/bin/env python3
"""Rotation invariance experiment for the SIFT scratch implementation.

Setup
-----
- img1 : the 500×500 source image directly (no rotation)
- img2 : the 500×500 source image rotated by θ about its centre

Evaluation region
-----------------
The evaluation region is a centred disk of radius EVAL_RADIUS inside every
500×500 crop. A disk is perfectly rotation-symmetric, so the *exact same*
image content is evaluated at every angle — a point inside the disk maps to
another point inside the disk under any rotation about the centre. This makes
the comparison completely fair across angles. With EVAL_RADIUS = 250 the disk
is the largest circle inscribed in the 500×500 crop (it touches all four
edges), maximising the evaluated area while staying rotation-symmetric.
Only matches whose both endpoints lie inside this disk are counted in the
statistics, and the area outside it is darkened in the visualisation.

Outputs
-------
  experiments/results/rotation/
    {stem}_angle{θ:03d}.png   per-angle visualisation
    summary.png               inlier-rate vs angle bar chart
    stats.json                full statistics (JSON)

Usage
-----
    uv run python experiments/rotation_experiment.py
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

# Allow running from the repo root or from inside experiments/
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

CROP_SIZE     = 500
EVAL_RADIUS   = 250     # inscribed disk of 500×500 crop; rotation-symmetric → fair across all angles
ANGLES        = [0, 15, 30, 60, 90, 120, 180]
IMAGE_NAMES   = ["source1.jpg"]

IMAGES_DIR    = Path(__file__).parent / "inputs"
OUT_DIR       = Path(__file__).parent / "outputs" / "rotation"

SIFT_OCTAVES  = 4
SIFT_SCALES   = 3
SIFT_SIGMA    = 1.6
RATIO         = 0.75
AFFINE_THRESH = 6.0

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate *img* (500×500) around its centre by *angle_deg* (CCW).

    Points within the inscribed disk (radius 250) always map to valid source
    pixels, so no black border appears inside the evaluation region.
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def compute_eval_mask(crop_size: int = CROP_SIZE,
                      eval_radius: int = EVAL_RADIUS) -> np.ndarray:
    """Boolean mask marking the central circular evaluation region of a crop.

    The same disk mask is used for *both* img1 and img2 regardless of θ.
    Because a disk centred at the crop centre is rotation-symmetric, every
    angle evaluates exactly the same image content — a point inside the disk
    stays inside the disk under any rotation about the centre.
    """
    half = crop_size / 2.0
    ys, xs = np.mgrid[0:crop_size, 0:crop_size]
    dist2 = (xs - half + 0.5) ** 2 + (ys - half + 0.5) ** 2
    return dist2 <= eval_radius ** 2


def kp_xy(kp: dict) -> tuple[float, float]:
    """(col_px, row_px) of keypoint in crop-image pixel coordinates."""
    s = 2.0 ** kp["octave"]
    return kp["col"] * s / 2.0, kp["row"] * s / 2.0


def kp_in_mask(kp: dict, mask: np.ndarray) -> bool:
    x, y = kp_xy(kp)
    r = int(np.clip(round(y), 0, mask.shape[0] - 1))
    c = int(np.clip(round(x), 0, mask.shape[1] - 1))
    return bool(mask[r, c])


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


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _apply_overlap_overlay(bgr: np.ndarray, mask: np.ndarray,
                            alpha: float = 0.25) -> np.ndarray:
    """Darken pixels outside the evaluation region."""
    out = bgr.copy().astype(np.float32)
    out[~mask] *= alpha
    return out.clip(0, 255).astype(np.uint8)


def save_match_figure(
    bgr1: np.ndarray,
    bgr2: np.ndarray,
    mask: np.ndarray,
    kps1: list[dict],
    kps2: list[dict],
    matches: list[tuple[int, int, float]],
    inlier_mask: np.ndarray,
    stats: dict,
    title: str,
    out_path: Path,
) -> None:
    img1_vis = cv2.cvtColor(_apply_overlap_overlay(bgr1, mask), cv2.COLOR_BGR2RGB)
    img2_vis = cv2.cvtColor(_apply_overlap_overlay(bgr2, mask), cv2.COLOR_BGR2RGB)

    h1, w1 = img1_vis.shape[:2]
    h2, w2 = img2_vis.shape[:2]
    canvas  = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_vis
    canvas[:h2, w1:] = img2_vis

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(canvas)

    for i, (i1, i2, _) in enumerate(matches):
        if not (kp_in_mask(kps1[i1], mask) and kp_in_mask(kps2[i2], mask)):
            continue
        x1, y1 = kp_xy(kps1[i1])
        x2, y2 = kp_xy(kps2[i2])
        x2 += w1
        is_inlier = bool(inlier_mask[i])
        color = "lime" if is_inlier else "red"
        alpha = 0.85 if is_inlier else 0.35
        lw    = 0.8  if is_inlier else 0.4
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha)
        ax.plot(x1, y1, ".", color=color, ms=3, alpha=alpha)
        ax.plot(x2, y2, ".", color=color, ms=3, alpha=alpha)

    ax.axis("off")
    ax.set_title(title, fontsize=9)

    n_eval = stats["n_matches_in_eval"] - stats["n_inliers_in_eval"]
    legend = [
        mpatches.Patch(color="lime",  label=f"Inlier  ({stats['n_inliers_in_eval']})"),
        mpatches.Patch(color="red",   label=f"Outlier ({n_eval})"),
        mpatches.Patch(facecolor="white", edgecolor="gray",
                       alpha=0.5, label=f"Outside r={EVAL_RADIUS} eval disk (darkened)"),
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
        angle_stats = all_stats[stem]
        angles      = [int(a) for a in angle_stats]
        rates       = [angle_stats[str(a)]["inlier_rate_in_eval"]  for a in angles]
        n_inliers   = [angle_stats[str(a)]["n_inliers_in_eval"]    for a in angles]
        n_matches   = [angle_stats[str(a)]["n_matches_in_eval"]    for a in angles]

        xs = range(len(angles))
        
        # Left axis: inlier rate
        ax.plot(xs, rates, marker="o", color=cmap(0), lw=1.8, ms=6, label="Inlier rate")
        ax.set_ylabel("Inlier rate (in eval region)", color=cmap(0), fontsize=10)
        ax.tick_params(axis="y", labelcolor=cmap(0))
        ax.set_ylim(0, 1.20)
        
        # Right axis: match count
        ax2 = ax.twinx()
        ax2.plot(xs, n_matches, marker="s", color=cmap(1), lw=1.8, ms=5, label="Match count")
        ax2.set_ylabel("Match count", color=cmap(1), fontsize=10)
        ax2.tick_params(axis="y", labelcolor=cmap(1))
        ax2.set_ylim(0, max(n_matches) * 1.1 if n_matches else 1)

        for i, (r, ni, nm) in enumerate(zip(rates, n_inliers, n_matches)):
            ax.text(i, r + 0.018, f"{ni}/{nm}",
                    ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{a}°" for a in angles])
        ax.set_xlabel("Rotation angle")
        ax.set_title(stem)
        ax.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.4)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    fig.suptitle(
        "SIFT rotation invariance",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary → {out_path}")


# ---------------------------------------------------------------------------
# Per-image experiment
# ---------------------------------------------------------------------------

def run_one_image(bgr_full: np.ndarray, stem: str) -> dict:
    angle_stats: dict = {}
    eval_mask = compute_eval_mask()

    for angle in ANGLES:
        print(f"    {angle:3d}°", end="  ", flush=True)
        t0 = time.perf_counter()

        # ── crops ─────────────────────────────────────────────────────────
        bgr1 = bgr_full
        bgr2 = rotate_image(bgr_full, angle)

        gray1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

        # ── SIFT ──────────────────────────────────────────────────────────
        kps1, descs1 = run_sift(gray1)
        kps2, descs2 = run_sift(gray2)

        # ── Match + RANSAC ────────────────────────────────────────────────
        matches = match_descriptors(descs1, descs2, ratio_threshold=RATIO)

        if len(matches) >= 4:
            _, inlier_mask = geometric_verification(
                kps1, kps2, matches, gray1.shape,
                affine_thresh=AFFINE_THRESH,
            )
        else:
            inlier_mask = np.zeros(len(matches), dtype=bool)

        # ── Filter to evaluation region ───────────────────────────────────
        n_kp1_ev = sum(kp_in_mask(k, eval_mask) for k in kps1)
        n_kp2_ev = sum(kp_in_mask(k, eval_mask) for k in kps2)

        ev_idx = [
            i for i, (i1, i2, _) in enumerate(matches)
            if kp_in_mask(kps1[i1], eval_mask) and kp_in_mask(kps2[i2], eval_mask)
        ]
        n_m_ev  = len(ev_idx)
        n_in_ev = int(sum(bool(inlier_mask[i]) for i in ev_idx))
        rate    = n_in_ev / max(n_m_ev, 1)

        elapsed = time.perf_counter() - t0
        print(
            f"kp={len(kps1)}+{len(kps2)}"
            f"  kp_ev={n_kp1_ev}+{n_kp2_ev}"
            f"  m_ev={n_m_ev}  inl_ev={n_in_ev}"
            f"  rate={rate:.1%}  [{elapsed:.1f}s]"
        )

        # ── Stats ─────────────────────────────────────────────────────────
        stats = {
            "angle":               angle,
            "crop_size":           CROP_SIZE,
            "eval_radius":         EVAL_RADIUS,
            "n_kp1":               len(kps1),
            "n_kp2":               len(kps2),
            "n_kp1_in_eval":       n_kp1_ev,
            "n_kp2_in_eval":       n_kp2_ev,
            "n_matches":           len(matches),
            "n_matches_in_eval":   n_m_ev,
            "n_inliers_in_eval":   n_in_ev,
            "inlier_rate_in_eval": round(rate, 4),
            "elapsed_sec":         round(elapsed, 2),
        }
        angle_stats[str(angle)] = stats

        # ── Visualise ─────────────────────────────────────────────────────
        title = (
            f"{stem}  |  θ = {angle}°  |  "
            f"eval disk: r={EVAL_RADIUS}  |  "
            f"kp in eval: {n_kp1_ev} + {n_kp2_ev}  |  "
            f"matches in eval: {n_m_ev}  |  "
            f"inliers in eval: {n_in_ev}  |  "
            f"inlier rate: {rate:.1%}"
        )
        save_match_figure(
            bgr1, bgr2, eval_mask,
            kps1, kps2, matches, inlier_mask,
            stats, title,
            OUT_DIR / f"{stem}_angle{angle:03d}.png",
        )

    return angle_stats


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

    # JSON stats
    stats_path = OUT_DIR / "rotation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats  → {stats_path}")

    # Summary figure
    save_summary_figure(all_stats, OUT_DIR / "rotation_summary.png")


if __name__ == "__main__":
    main()
