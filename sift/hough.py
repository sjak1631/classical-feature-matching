"""Hough-voting geometric verification and affine pose estimation.

Implements Lowe (2004) §5 (pose clustering) and affine fitting:

- 4-D Hough accumulator: (orientation, log2-scale, x-position, y-position)
- Bin widths (paper table):
    orientation  : 30°
    scale        : factor-of-2 (log2 bins, width = 1)
    position     : 0.25 × predicted_scale × max(img1_h, img1_w)
- Each match votes in the nearest 2 bins per dimension → 2⁴ = 16 Hough cells
- Clusters with ≥ 3 matches are accepted as pose hypotheses  (paper: §5)
- Each cluster undergoes iterative affine least-squares fitting  (paper: §6)

This replaces the RANSAC homography used in earlier versions.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Lowe (2004) constants
# ---------------------------------------------------------------------------

ORIENT_BIN_RAD   = np.deg2rad(30.0)   # orientation bin width (§5 table: 30°)
SCALE_BIN_LOG2   = 1.0                 # scale bin width in log2 (factor-of-2)
POS_BIN_FRACTION = 0.25                # position bin = fraction × scale × max_dim  (§5 table)
MIN_CLUSTER      = 3                   # minimum correspondences per cluster  (§5)

# Affine inlier threshold – paper does not specify; 6 px is an implementation choice.
AFFINE_THRESH    = 6.0
_MAX_REFINE_ITER = 5                   # max affine re-fit iterations


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kp_xy(kp: dict) -> tuple[float, float]:
    """Image-plane coordinates, accounting for the Lowe 2× upsampling."""
    s = 2.0 ** kp["octave"]
    return kp["col"] * s / 2.0, kp["row"] * s / 2.0


def _fit_affine(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Least-squares 2-D affine fit:  pts2 ≈ M @ pts1^T + t.

    Solves the 6-parameter system [m1,m2,tx,m3,m4,ty] via numpy lstsq.
    Returns (M[2×2], t[2]) or None if underdetermined.
    """
    n = len(pts1)
    if n < 3:
        return None

    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    for i, ((x, y), (u, v)) in enumerate(zip(pts1, pts2)):
        A[2 * i,   :3] = [x, y, 1.0]
        A[2 * i + 1, 3:] = [x, y, 1.0]
        b[2 * i]     = u
        b[2 * i + 1] = v

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    M = np.array([[coeffs[0], coeffs[1]],
                  [coeffs[3], coeffs[4]]], dtype=np.float64)
    t = np.array([coeffs[2], coeffs[5]], dtype=np.float64)
    return M, t


def _affine_residuals(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    M: NDArray[np.float64],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-point L2 residual under the affine map  pts2 ≈ M @ pts1^T + t."""
    pred = (M @ pts1.T).T + t   # (N, 2)
    return np.sqrt(((pred - pts2) ** 2).sum(axis=1))


# ---------------------------------------------------------------------------
# Hough accumulation
# ---------------------------------------------------------------------------

def _hough_accumulate(
    kps1: list[dict],
    kps2: list[dict],
    matches: list[tuple[int, int, float]],
    max_dim: int,
) -> dict[tuple[int, int, int, int], list[int]]:
    """Build 4-D Hough accumulator.

    Each match casts 2⁴ = 16 votes (nearest 2 bins in each dimension).
    Returns {(ori_bin, scale_bin, px_bin, py_bin) → [match_indices]}.
    """
    accumulator: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)

    for m_idx, (i1, i2, _) in enumerate(matches):
        kp1, kp2 = kps1[i1], kps2[i2]
        x1, y1 = _kp_xy(kp1)
        x2, y2 = _kp_xy(kp2)

        sigma1 = kp1["sigma"]
        sigma2 = kp2["sigma"]
        if sigma1 < 1e-12:
            continue
        scale_ratio = sigma2 / sigma1

        angle_diff = (kp2["angle"] - kp1["angle"]) % (2.0 * np.pi)

        # Predicted similarity transform:  x2 ≈ s·R(θ)·x1 + t
        cos_a = np.cos(angle_diff)
        sin_a = np.sin(angle_diff)
        tx = x2 - scale_ratio * (cos_a * x1 - sin_a * y1)
        ty = y2 - scale_ratio * (sin_a * x1 + cos_a * y1)

        # Continuous bin coordinates
        ori_f   = angle_diff / ORIENT_BIN_RAD               # [0, 12)
        scale_f = np.log2(scale_ratio) / SCALE_BIN_LOG2     # log2 bins

        # Position bin width: Lowe (2004) §5 — 0.25 × scale_ratio × max_dim
        pos_bin_w = max(POS_BIN_FRACTION * scale_ratio * max_dim, 1.0)
        px_f = tx / pos_bin_w
        py_f = ty / pos_bin_w

        # Vote in nearest 2 bins in each of 4 dimensions → 16 Hough cells
        ori_b0   = int(np.floor(ori_f))
        scale_b0 = int(np.floor(scale_f))
        px_b0    = int(np.floor(px_f))
        py_b0    = int(np.floor(py_f))

        for d_ori in range(2):
            for d_sc in range(2):
                for d_px in range(2):
                    for d_py in range(2):
                        key = (
                            ori_b0   + d_ori,
                            scale_b0 + d_sc,
                            px_b0   + d_px,
                            py_b0   + d_py,
                        )
                        accumulator[key].append(m_idx)

    return dict(accumulator)


# ---------------------------------------------------------------------------
# Affine verification for one cluster
# ---------------------------------------------------------------------------

def _verify_cluster(
    pts1: NDArray[np.float64],
    pts2: NDArray[np.float64],
    cluster_idx: list[int],
    threshold: float,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, list[int]]:
    """Fit affine to a Hough cluster; iteratively extend and refine.

    Args:
        pts1, pts2: All matched image coordinates (one row per match).
        cluster_idx: Initial match indices from the Hough bin (may have duplicates).
        threshold: Affine inlier pixel threshold.

    Returns:
        (M, t, inlier_indices) where inlier_indices indexes into pts1/pts2.
    """
    idx = list(set(cluster_idx))   # deduplicate

    for _ in range(_MAX_REFINE_ITER):
        result = _fit_affine(pts1[idx], pts2[idx])
        if result is None:
            return None, None, []
        M, t = result

        # Evaluate all matches and expand inlier set
        resid   = _affine_residuals(pts1, pts2, M, t)
        new_idx = [i for i, r in enumerate(resid) if r < threshold]

        if len(new_idx) < 3 or set(new_idx) == set(idx):
            break
        idx = new_idx

    return M, t, idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def geometric_verification(
    kps1: list[dict],
    kps2: list[dict],
    matches: list[tuple[int, int, float]],
    img1_shape: tuple[int, int],
    min_cluster: int = MIN_CLUSTER,
    affine_thresh: float = AFFINE_THRESH,
) -> tuple[tuple[NDArray, NDArray] | None, NDArray[np.bool_]]:
    """Hough-vote clustering + affine pose estimation (Lowe 2004 §5–§6).

    Drop-in replacement for the former ransac_homography call.

    Args:
        kps1: Oriented keypoints from image 1.
        kps2: Oriented keypoints from image 2.
        matches: (idx1, idx2, dist) triples from match_descriptors.
        img1_shape: (height, width) of image 1 (used for position bin width).
        min_cluster: Minimum cluster size to accept (paper: 3).
        affine_thresh: Affine inlier pixel threshold (paper unspecified; 6 px).

    Returns:
        ((M[2×2], t[2]) | None, inlier_mask) where inlier_mask is bool
        over *matches* indicating geometrically consistent correspondences.
    """
    n_matches = len(matches)
    inlier_mask = np.zeros(n_matches, dtype=bool)

    if n_matches < min_cluster:
        return None, inlier_mask

    max_dim = max(img1_shape)

    # Build image-coordinate arrays (one row per match)
    pts1 = np.array([_kp_xy(kps1[i1]) for i1, _, _ in matches], dtype=np.float64)
    pts2 = np.array([_kp_xy(kps2[i2]) for _, i2, _ in matches], dtype=np.float64)

    # 4-D Hough accumulation
    accumulator = _hough_accumulate(kps1, kps2, matches, max_dim)

    # Collect clusters with ≥ min_cluster unique matches
    clusters = [
        list(set(cell_idx))
        for cell_idx in accumulator.values()
        if len(set(cell_idx)) >= min_cluster
    ]

    if not clusters:
        return None, inlier_mask

    # Verify each cluster; keep the one with the most affine inliers
    best_affine: tuple[NDArray, NDArray] | None = None
    best_inliers: list[int] = []

    for cluster in clusters:
        M, t, inliers = _verify_cluster(pts1, pts2, cluster, affine_thresh)
        if M is not None and len(inliers) > len(best_inliers):
            best_affine = (M, t)
            best_inliers = inliers

    for i in best_inliers:
        inlier_mask[i] = True

    return best_affine, inlier_mask
