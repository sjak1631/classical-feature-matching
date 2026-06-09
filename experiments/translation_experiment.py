#!/usr/bin/env python3
"""Translation invariance experiment for the SIFT scratch implementation.

Setup
-----
- img1 : the 500×500 source image (no translation)
- img2 : img1 translated by d pixels in a cardinal direction via cv2.warpAffine

Evaluation region
-----------------
The evaluation region for img1 is the central 300×300 window
(rows/cols 100–399 in a 500×500 image).  For any |translation| ≤ 100 px,
this window lies entirely within the overlap between img1 and img2.  The
corresponding window in img2 is the same 300×300 rectangle shifted by the
translation vector (tx, ty).

    img1 eval window : rows [100, 399], cols [100, 399]
    img2 eval window : rows [100+ty, 399+ty], cols [100+tx, 399+tx]

Only matches whose img1 keypoint lies in the img1 window *and* whose img2
keypoint lies in the img2 window are counted in the statistics.  Both
windows are highlighted (outside region darkened) in every visualisation.

Summary statistics average the four cardinal directions (right, left, up,
down) for each translation amount d.

Outputs
-------
  experiments/results/translation/
    {stem}_{direction}_d{d:03d}.png   per-(direction, amount) visualisation
    summary.png                        inlier-rate vs amount (avg over dirs)
    stats.json                         full statistics (JSON)

Usage
-----
    uv run python experiments/translation_experiment.py
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

IMG_SIZE     = 500
EVAL_MARGIN  = 100          # (IMG_SIZE - EVAL_SIZE) // 2
EVAL_SIZE    = 300          # central 300×300 evaluation window

TRANSLATIONS: list[int] = [0, 25, 50, 75, 100]

DIRECTIONS: list[tuple[str, int, int]] = [
    ("right",  1,  0),
    ("left",  -1,  0),
    ("up",     0, -1),
    ("down",   0,  1),
]

IMAGE_NAMES  = ["source1.jpg"]
IMAGES_DIR   = Path(__file__).parent / "inputs"
OUT_DIR      = Path(__file__).parent / "outputs" / "translation"

SIFT_OCTAVES  = 4
SIFT_SCALES   = 3
SIFT_SIGMA    = 1.6
RATIO         = 0.75
AFFINE_THRESH = 6.0

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def translate_image(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    """Return *img* translated by (tx, ty) pixels.

    Positive tx → content shifts right; positive ty → content shifts down.
    Vacated border is filled with black.
    """
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def eval_masks(tx: int, ty: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask1, mask2): boolean IMG_SIZE×IMG_SIZE evaluation masks.

    mask1 — central EVAL_SIZE×EVAL_SIZE of img1.
    mask2 — the same rectangle shifted by (tx, ty) in img2,
             clamped to the image boundary.

    For |tx|, |ty| ≤ EVAL_MARGIN the shifted rectangle fits entirely within
    the image boundary (no clamping needed).
    """
    m1 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
    m1[EVAL_MARGIN : IMG_SIZE - EVAL_MARGIN,
       EVAL_MARGIN : IMG_SIZE - EVAL_MARGIN] = True

    m2 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
    r0 = max(EVAL_MARGIN + ty, 0)
    c0 = max(EVAL_MARGIN + tx, 0)
    r1 = min(IMG_SIZE - EVAL_MARGIN + ty, IMG_SIZE)
    c1 = min(IMG_SIZE - EVAL_MARGIN + tx, IMG_SIZE)
    if r1 > r0 and c1 > c0:
        m2[r0:r1, c0:c1] = True

    return m1, m2


def kp_xy(kp: dict) -> tuple[float, float]:
    """(col_px, row_px) of keypoint in image pixel coordinates."""
    s = 2.0 ** kp["octave"]
    return kp["col"] * s / 2.0, kp["row"] * s / 2.0


def kp_in_mask(kp: dict, mask: np.ndarray) -> bool:
    x, y = kp_xy(kp)
    r = int(np.clip(round(y), 0, mask.shape[0] - 1))
    c = int(np.clip(round(x), 0, mask.shape[1] - 1))
    return bool(mask[r, c])


def _mask_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (r0, c0, r1, c1) of True pixels, or None if mask is empty."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


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

def _apply_mask_overlay(bgr: np.ndarray, mask: np.ndarray,
                        alpha: float = 0.25) -> np.ndarray:
    """Darken pixels outside the evaluation region."""
    out = bgr.copy().astype(np.float32)
    out[~mask] *= alpha
    return out.clip(0, 255).astype(np.uint8)


def save_match_figure(
    bgr1: np.ndarray,
    bgr2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    kps1: list[dict],
    kps2: list[dict],
    matches: list[tuple[int, int, float]],
    inlier_mask: np.ndarray,
    stats: dict,
    title: str,
    out_path: Path,
) -> None:
    img1_vis = cv2.cvtColor(_apply_mask_overlay(bgr1, mask1), cv2.COLOR_BGR2RGB)
    img2_vis = cv2.cvtColor(_apply_mask_overlay(bgr2, mask2), cv2.COLOR_BGR2RGB)

    h1, w1 = img1_vis.shape[:2]
    h2, w2 = img2_vis.shape[:2]
    canvas  = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_vis
    canvas[:h2, w1:] = img2_vis

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(canvas)

    # Match lines (eval-region pairs only)
    for i, (i1, i2, _) in enumerate(matches):
        if not (kp_in_mask(kps1[i1], mask1) and kp_in_mask(kps2[i2], mask2)):
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

    # Yellow dashed rectangles showing the evaluation windows
    for mask, x_offset in [(mask1, 0), (mask2, w1)]:
        rect = _mask_rect(mask)
        if rect is not None:
            r0, c0, r1, c1 = rect
            ax.add_patch(mpatches.Rectangle(
                (c0 + x_offset, r0),
                c1 - c0 + 1, r1 - r0 + 1,
                linewidth=1.5, edgecolor="yellow",
                facecolor="none", linestyle="--",
            ))

    ax.axis("off")
    ax.set_title(title, fontsize=9)

    n_outlier = stats["n_matches_in_eval"] - stats["n_inliers_in_eval"]
    legend = [
        mpatches.Patch(color="lime", label=f"Inlier  ({stats['n_inliers_in_eval']})"),
        mpatches.Patch(color="red",  label=f"Outlier ({n_outlier})"),
        mpatches.Patch(facecolor="none", edgecolor="yellow", linestyle="--",
                       linewidth=1.5, label=f"Eval region ({EVAL_SIZE}×{EVAL_SIZE})"),
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
        ds = all_stats[stem]
        avg_rates, std_rates = [], []
        avg_inl,   avg_mat   = [], []

        for d in TRANSLATIONS:
            rates, inls, mats = [], [], []
            for dir_name, *_ in DIRECTIONS:
                key = f"{dir_name}_d{d:03d}"
                if key in ds:
                    rates.append(ds[key]["inlier_rate_in_eval"])
                    inls.append(ds[key]["n_inliers_in_eval"])
                    mats.append(ds[key]["n_matches_in_eval"])
            avg_rates.append(float(np.mean(rates)) if rates else 0.0)
            std_rates.append(float(np.std(rates))  if rates else 0.0)
            avg_inl.append(float(np.mean(inls))    if inls  else 0.0)
            avg_mat.append(float(np.mean(mats))    if mats  else 0.0)

        xs = range(len(TRANSLATIONS))
        
        # Left axis: inlier rate
        ax.errorbar(xs, avg_rates, yerr=std_rates,
                    marker="o", color=cmap(0), lw=1.8, ms=6,
                    capsize=4, elinewidth=1.0, alpha=0.8, label="Inlier rate")
        ax.set_ylabel("Inlier rate (avg over 4 directions)", color=cmap(0), fontsize=10)
        ax.tick_params(axis="y", labelcolor=cmap(0))
        ax.set_ylim(0, 1.25)
        
        # Right axis: match count
        ax2 = ax.twinx()
        ax2.plot(xs, avg_mat, marker="s", color=cmap(1), lw=1.8, ms=5, label="Match count")
        ax2.set_ylabel("Match count (avg over 4 directions)", color=cmap(1), fontsize=10)
        ax2.tick_params(axis="y", labelcolor=cmap(1))
        ax2.set_ylim(0, max(avg_mat) * 1.1 if avg_mat else 1)

        for i, (r, e, ni, nm) in enumerate(zip(avg_rates, std_rates, avg_inl, avg_mat)):
            ax.text(i, r + e + 0.015,
                    f"{ni:.0f}/{nm:.0f}",
                    ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{d}px" for d in TRANSLATIONS])
        ax.set_xlabel("Translation amount")
        ax.set_title(stem)
        ax.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.4)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    fig.suptitle(
        "SIFT translation invariance",
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
    direction_stats: dict = {}

    # img1 is the fixed reference — compute SIFT once
    gray1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    kps1, descs1 = run_sift(gray1)

    for dir_name, dx, dy in DIRECTIONS:
        print(f"  [{dir_name}]")
        for d in TRANSLATIONS:
            tx = dx * d
            ty = dy * d
            key = f"{dir_name}_d{d:03d}"
            print(f"    d={d:3d}px", end="  ", flush=True)
            t0 = time.perf_counter()

            # ── Images ───────────────────────────────────────────────────────
            bgr2         = translate_image(bgr, tx, ty)
            mask1, mask2 = eval_masks(tx, ty)

            if tx == 0 and ty == 0:
                # d=0: img2 identical to img1 → reuse SIFT results
                kps2, descs2 = kps1, descs1
            else:
                gray2 = (cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
                         .astype(np.float64) / 255.0)
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
            n_kp1_ev = sum(kp_in_mask(k, mask1) for k in kps1)
            n_kp2_ev = sum(kp_in_mask(k, mask2) for k in kps2)

            ev_idx = [
                i for i, (i1, i2, _) in enumerate(matches)
                if kp_in_mask(kps1[i1], mask1) and kp_in_mask(kps2[i2], mask2)
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
            direction_stats[key] = {
                "direction":           dir_name,
                "tx":                  tx,
                "ty":                  ty,
                "img_size":            IMG_SIZE,
                "eval_size":           EVAL_SIZE,
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

            # ── Visualise ─────────────────────────────────────────────────────
            title = (
                f"{stem}  |  {dir_name} d={d}px (tx={tx}, ty={ty})  |  "
                f"eval {EVAL_SIZE}×{EVAL_SIZE}  |  "
                f"kp_ev: {n_kp1_ev}+{n_kp2_ev}  |  "
                f"m_ev: {n_m_ev}  |  inl_ev: {n_in_ev}  |  rate: {rate:.1%}"
            )
            save_match_figure(
                bgr, bgr2, mask1, mask2,
                kps1, kps2, matches, inlier_mask,
                direction_stats[key], title,
                OUT_DIR / f"{stem}_{dir_name}_d{d:03d}.png",
            )

    return direction_stats


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

    stats_path = OUT_DIR / "translation_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats  → {stats_path}")

    save_summary_figure(all_stats, OUT_DIR / "translation_summary.png")


if __name__ == "__main__":
    main()
