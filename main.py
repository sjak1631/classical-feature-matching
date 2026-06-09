"""SIFT feature matching demo.

Usage
-----
    uv run python main.py <image1> <image2> [options]

Examples
--------
    uv run python main.py images/left.jpg images/right.jpg
    uv run python main.py images/left.jpg images/right.jpg --save matches.png
    uv run python main.py images/left.jpg images/right.jpg --octaves 4 --scales 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and return (bgr, gray_float64 in [0,1])."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    return bgr, gray


def run_sift(
    gray: np.ndarray,
    num_octaves: int,
    num_scales: int,
    sigma: float,
) -> tuple[list[dict], np.ndarray]:
    """Full SIFT pipeline: returns (oriented_keypoints, descriptors)."""
    gp   = build_gaussian_pyramid(gray, num_octaves=num_octaves, num_scales=num_scales, sigma=sigma)
    dp   = build_dog_pyramid(gp)
    cands = detect_keypoints(dp)
    kps  = localize_keypoints(dp, cands)
    kps  = assign_orientations(gp, kps)
    descs = compute_descriptors(gp, kps)
    return kps, descs


def kp_to_image_coords(kps: list[dict]) -> np.ndarray:
    """Convert keypoint dicts to (N, 2) pixel coordinates in the original image."""
    coords = np.array(
        [[kp["col"] * (2.0 ** kp["octave"]) / 2.0,
          kp["row"] * (2.0 ** kp["octave"]) / 2.0] for kp in kps],
        dtype=np.float64,
    )
    return coords


def draw_matches(
    bgr1: np.ndarray,
    bgr2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inlier_mask: np.ndarray,
) -> np.ndarray:
    """Draw inlier (green) and outlier (red) matches side by side."""
    h1, w1 = bgr1.shape[:2]
    h2, w2 = bgr2.shape[:2]
    h_out = max(h1, h2)
    canvas = np.zeros((h_out, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = bgr1
    canvas[:h2, w1:] = bgr2

    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        color = (0, 200, 0) if inlier_mask[i] else (0, 0, 200)
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])) + w1, int(round(p2[1]))
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x2, y2), 3, color, -1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SIFT scratch implementation — feature matching demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image1", help="Path to first image")
    parser.add_argument("image2", help="Path to second image")
    parser.add_argument("--octaves",  type=int,   default=4,    help="Number of octaves")
    parser.add_argument("--scales",   type=int,   default=3,    help="Scales per octave (s)")
    parser.add_argument("--sigma",    type=float, default=1.6,  help="Base Gaussian sigma")
    parser.add_argument("--ratio",    type=float, default=0.8,  help="Lowe ratio test threshold")
    parser.add_argument("--threshold",type=float, default=6.0,  help="Affine inlier threshold (px)")
    parser.add_argument("--save",     default=None,             help="Save match image to this path")
    parser.add_argument("--no-show",  action="store_true",      help="Do not display the result window")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"Loading images...")
    bgr1, gray1 = load_image(args.image1)
    bgr2, gray2 = load_image(args.image2)
    print(f"  Image 1: {args.image1}  {gray1.shape[1]}×{gray1.shape[0]}")
    print(f"  Image 2: {args.image2}  {gray2.shape[1]}×{gray2.shape[0]}")

    # ------------------------------------------------------------------
    # SIFT pipeline
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print(f"\nRunning SIFT (octaves={args.octaves}, scales={args.scales}, σ={args.sigma})...")

    kps1, descs1 = run_sift(gray1, args.octaves, args.scales, args.sigma)
    kps2, descs2 = run_sift(gray2, args.octaves, args.scales, args.sigma)
    t_sift = time.perf_counter() - t0

    print(f"  Keypoints: {len(kps1)} (image 1)  {len(kps2)} (image 2)  [{t_sift:.2f}s]")

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    matches = []
    affine = None
    inlier_mask = np.array([], dtype=bool)
    
    if len(kps1) > 0 and len(kps2) > 0:
        t1 = time.perf_counter()
        matches = match_descriptors(descs1, descs2, ratio_threshold=args.ratio)
        t_match = time.perf_counter() - t1
        print(f"  Matches after ratio test: {len(matches)}  [{t_match:.3f}s]")

        if len(matches) >= 3:
            # ------------------------------------------------------------------
            # Hough voting + affine verification  (Lowe 2004 §5–§6)
            # ------------------------------------------------------------------
            t2 = time.perf_counter()
            affine, inlier_mask = geometric_verification(
                kps1, kps2, matches, gray1.shape,
                affine_thresh=args.threshold,
            )
            t_hough = time.perf_counter() - t2

            n_inliers = int(inlier_mask.sum())
            print(f"  Hough inliers: {n_inliers}/{len(matches)}  [{t_hough:.3f}s]")
            if affine is not None:
                M, t = affine
                print(f"  Affine M:\n{np.array2string(M, precision=4, suppress_small=True)}")
                print(f"  Affine t: {np.array2string(t, precision=4, suppress_small=True)}")
            else:
                print("  Hough verification found no consistent pose cluster.")
        else:
            print("  Too few matches for geometric verification.")
            inlier_mask = np.zeros(len(matches), dtype=bool)
    else:
        print("  No keypoints found in one or both images.")

    # Prepare matched points for visualization
    idx1 = np.array([m[0] for m in matches]) if len(matches) > 0 else np.array([], dtype=int)
    idx2 = np.array([m[1] for m in matches]) if len(matches) > 0 else np.array([], dtype=int)

    pts1_full = kp_to_image_coords(kps1)
    pts2_full = kp_to_image_coords(kps2)
    matched_pts1 = pts1_full[idx1] if len(idx1) > 0 else np.zeros((0, 2), dtype=np.float32)
    matched_pts2 = pts2_full[idx2] if len(idx2) > 0 else np.zeros((0, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    canvas = draw_matches(bgr1, bgr2, matched_pts1, matched_pts2, inlier_mask)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.imshow(canvas_rgb)
    ax.axis("off")
    
    n_inliers = int(inlier_mask.sum()) if len(inlier_mask) > 0 else 0
    n_outliers = len(matches) - n_inliers
    
    title = (
        f"SIFT matching  |  kp: {len(kps1)} + {len(kps2)}"
        f"  |  matches: {len(matches)}"
        f"  |  inliers: {n_inliers}"
    )
    ax.set_title(title)
    legend = [
        mpatches.Patch(color="green", label=f"Inliers ({n_inliers})"),
        mpatches.Patch(color="red",   label=f"Outliers ({n_outliers})"),
    ]
    ax.legend(handles=legend, loc="upper right")
    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nSaved to: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

