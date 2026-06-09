#!/usr/bin/env python3
"""Scale invariance experiment for the SIFT scratch implementation.

Setup
-----
- img1 : the 500×500 source image (scale = 1.0, reference)
- img2 : img1 scaled by *s*, keeping the output at 500×500:
    s > 1  (zoom in)  → resize img1 to 500·s × 500·s, then centre-crop 500×500.
                        img2 shows the centre 500/s × 500/s content of img1,
                        magnified s×.
    s < 1  (zoom out) → resize img1 to 500·s × 500·s, then pad with black to
                        500×500.  img2 shows all of img1's content at s×
                        resolution, centred on a black canvas.

Evaluation region
-----------------
To compare the *same original content* across every scale factor, the
evaluated region in img1 is FIXED: the central R×R square, where

    R = round(500 / max(SCALES))            ← largest square visible at every s

This is the biggest content window that stays inside img2 even at the most
extreme zoom-in factor max(SCALES).  Under the centred similarity
    img2 = centre + (img1 - centre) · s
that same content occupies the central (R·s)×(R·s) square of img2:

    eval_size1 = R                          ← fixed for all s (same content)
    eval_size2 = min(500, round(R * s))     ← where that content lands in img2

Both regions are centred.  For SCALES with max 2.0 (→ R=250):
    s=2.0: eval 250×250 in img1, 500×500 in img2.
    s=1.0: eval 250×250 in img1, 250×250 in img2.
    s=0.5: eval 250×250 in img1, 125×125 in img2.
The img1 window — hence the compared content — is identical in all cases.

Outputs
-------
  experiments/outputs/scale/
    {stem}_scale{scale_str}.png   per-scale visualisation
    summary.png                   inlier-rate vs scale bar chart
    stats.json                    full statistics (JSON)

Usage
-----
    uv run python experiments/scale_experiment.py
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

IMG_SIZE = 500
SCALES: list[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# Fixed content window shared by *all* scale conditions: the central square of
# img1 that stays visible even at the largest zoom-in factor max(SCALES).
# Keeping img1's evaluation window constant guarantees every scale factor is
# scored on the exact same original content.
EVAL_CONTENT = round(IMG_SIZE / max(SCALES))

IMAGE_NAMES = ["source1.jpg"]
IMAGES_DIR  = Path(__file__).parent.parent / "experiments" / "inputs"
OUT_DIR     = Path(__file__).parent / "outputs" / "scale"

SIFT_OCTAVES  = 4
SIFT_SCALES   = 3
SIFT_SIGMA    = 1.6
RATIO         = 0.75
AFFINE_THRESH = 6.0


def _scale_str(s: float) -> str:
    """e.g. 0.5 → '050', 1.0 → '100', 2.0 → '200'"""
    return f"{int(round(s * 100)):03d}"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def scale_image(img: np.ndarray, s: float) -> np.ndarray:
    """Return *img* (500×500) scaled by *s*, output kept at 500×500.

    s > 1: enlarge then centre-crop → content is zoomed in.
    s < 1: shrink then centre-pad with black → content is zoomed out.
    """
    h, w = img.shape[:2]
    new_w, new_h = round(w * s), round(h * s)
    interp = cv2.INTER_LINEAR if s >= 1 else cv2.INTER_AREA
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    if s >= 1:
        # Crop centre back to IMG_SIZE × IMG_SIZE
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        return resized[y0 : y0 + h, x0 : x0 + w]
    else:
        # Pad with black to IMG_SIZE × IMG_SIZE
        out = np.zeros_like(img)
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return out


def eval_masks(s: float, img_size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask1, mask2): centred rectangular evaluation masks.

    The img1 mask is fixed (central EVAL_CONTENT×EVAL_CONTENT square) so every
    scale factor scores the *same* original content.  The img2 mask is that
    same content mapped through the centred similarity (central (R·s)×(R·s)).
    """
    size1 = EVAL_CONTENT
    size2 = min(img_size, round(EVAL_CONTENT * s))

    mask1 = np.zeros((img_size, img_size), dtype=bool)
    m1 = (img_size - size1) // 2
    mask1[m1 : m1 + size1, m1 : m1 + size1] = True

    mask2 = np.zeros((img_size, img_size), dtype=bool)
    m2 = (img_size - size2) // 2
    mask2[m2 : m2 + size2, m2 : m2 + size2] = True

    return mask1, mask2


def kp_xy(kp: dict) -> tuple[float, float]:
    s_px = 2.0 ** kp["octave"]
    return kp["col"] * s_px / 2.0, kp["row"] * s_px / 2.0


def kp_in_mask(kp: dict, mask: np.ndarray) -> bool:
    x, y = kp_xy(kp)
    r = int(np.clip(round(y), 0, mask.shape[0] - 1))
    c = int(np.clip(round(x), 0, mask.shape[1] - 1))
    return bool(mask[r, c])


def _mask_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
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

    for i, (i1, i2, _) in enumerate(matches):
        if not (kp_in_mask(kps1[i1], mask1) and kp_in_mask(kps2[i2], mask2)):
            continue
        x1, y1 = kp_xy(kps1[i1])
        x2, y2 = kp_xy(kps2[i2])
        x2 += w1
        is_inlier = bool(inlier_mask[i])
        color = "lime" if is_inlier else "red"
        av    = 0.85 if is_inlier else 0.35
        lw    = 0.8  if is_inlier else 0.4
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=av)
        ax.plot(x1, y1, ".", color=color, ms=3, alpha=av)
        ax.plot(x2, y2, ".", color=color, ms=3, alpha=av)

    # Yellow dashed rectangles showing evaluation windows
    for mask, x_off in [(mask1, 0), (mask2, w1)]:
        rect = _mask_rect(mask)
        if rect is not None:
            r0, c0, r1, c1 = rect
            ax.add_patch(mpatches.Rectangle(
                (c0 + x_off, r0), c1 - c0 + 1, r1 - r0 + 1,
                linewidth=1.5, edgecolor="yellow",
                facecolor="none", linestyle="--",
            ))

    ax.axis("off")
    ax.set_title(title, fontsize=9)

    s   = stats["scale"]
    s1  = stats["eval_size_img1"]
    s2  = stats["eval_size_img2"]
    n_out = stats["n_matches_in_eval"] - stats["n_inliers_in_eval"]
    legend = [
        mpatches.Patch(color="lime", label=f"Inlier  ({stats['n_inliers_in_eval']})"),
        mpatches.Patch(color="red",  label=f"Outlier ({n_out})"),
        mpatches.Patch(facecolor="none", edgecolor="yellow", linestyle="--",
                       linewidth=1.5,
                       label=f"Eval region  img1:{s1}² / img2:{s2}²"),
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
        rates   = [ds[_scale_str(s)]["inlier_rate_in_eval"]  for s in SCALES]
        inliers = [ds[_scale_str(s)]["n_inliers_in_eval"]    for s in SCALES]
        matches = [ds[_scale_str(s)]["n_matches_in_eval"]    for s in SCALES]

        xs = range(len(SCALES))
        
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

        for i, (r, ni, nm) in enumerate(zip(rates, inliers, matches)):
            ax.text(i, r + 0.018, f"{ni}/{nm}",
                    ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(list(xs))
        ax.set_xticklabels([str(s) for s in SCALES])
        ax.set_xlabel("Scale factor s")
        ax.set_title(stem)
        ax.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.4)
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")

    fig.suptitle(
        "SIFT scale invariance",
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
    scale_stats: dict = {}

    gray1    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    kps1_ref, descs1_ref = run_sift(gray1)

    for s in SCALES:
        key = _scale_str(s)
        print(f"  s={s:.2f}", end="  ", flush=True)
        t0 = time.perf_counter()

        bgr2         = scale_image(bgr, s)
        mask1, mask2 = eval_masks(s)

        if s == 1.0:
            kps2, descs2 = kps1_ref, descs1_ref
        else:
            gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
            kps2, descs2 = run_sift(gray2)

        kps1 = kps1_ref

        # ── Match + RANSAC ────────────────────────────────────────────────
        matches = match_descriptors(descs1_ref, descs2, ratio_threshold=RATIO)

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

        size1 = EVAL_CONTENT
        size2 = min(IMG_SIZE, round(EVAL_CONTENT * s))

        elapsed = time.perf_counter() - t0
        print(
            f"kp={len(kps1)}+{len(kps2)}"
            f"  kp_ev={n_kp1_ev}+{n_kp2_ev}"
            f"  m_ev={n_m_ev}  inl_ev={n_in_ev}"
            f"  rate={rate:.1%}  [{elapsed:.1f}s]"
        )

        scale_stats[key] = {
            "scale":               s,
            "eval_size_img1":      size1,
            "eval_size_img2":      size2,
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

        title = (
            f"{stem}  |  s = {s}  |  "
            f"eval img1:{size1}² img2:{size2}²  |  "
            f"kp_ev: {n_kp1_ev}+{n_kp2_ev}  |  "
            f"m_ev: {n_m_ev}  |  inl_ev: {n_in_ev}  |  rate: {rate:.1%}"
        )
        save_match_figure(
            bgr, bgr2, mask1, mask2,
            kps1, kps2, matches, inlier_mask,
            scale_stats[key], title,
            OUT_DIR / f"{stem}_scale{key}.png",
        )

    return scale_stats


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

    stats_path = OUT_DIR / "scale_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats  → {stats_path}")

    save_summary_figure(all_stats, OUT_DIR / "scale_summary.png")


if __name__ == "__main__":
    main()
