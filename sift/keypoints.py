"""
Keypoint detection and sub-pixel / sub-scale localization.

Implements:
- Local extremum detection in DoG pyramid
- Sub-pixel localization via quadratic interpolation (Taylor expansion)
- Contrast / edge response thresholding
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Lowe (2004) recommended defaults
CONTRAST_THRESHOLD = 0.03
EDGE_THRESHOLD = 10.0

_MAX_INTERP_ITER = 5
_BASE_SIGMA = 1.6


# ---------------------------------------------------------------------------
# Vectorised helpers for 26-neighbour extremum detection
# ---------------------------------------------------------------------------

def _layer_max(img: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise maximum of all 9 positions in a 3×3 spatial neighbourhood."""
    return np.maximum.reduce([
        img[:-2, :-2], img[:-2, 1:-1], img[:-2, 2:],
        img[1:-1, :-2], img[1:-1, 1:-1], img[1:-1, 2:],
        img[2:, :-2],  img[2:, 1:-1],  img[2:, 2:],
    ])


def _layer_min(img: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise minimum of all 9 positions in a 3×3 spatial neighbourhood."""
    return np.minimum.reduce([
        img[:-2, :-2], img[:-2, 1:-1], img[:-2, 2:],
        img[1:-1, :-2], img[1:-1, 1:-1], img[1:-1, 2:],
        img[2:, :-2],  img[2:, 1:-1],  img[2:, 2:],
    ])


def _spatial_max_excl_center(img: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise maximum of the 8 spatial neighbours, excluding centre."""
    return np.maximum.reduce([
        img[:-2, :-2], img[:-2, 1:-1], img[:-2, 2:],
        img[1:-1, :-2],                img[1:-1, 2:],
        img[2:, :-2],  img[2:, 1:-1],  img[2:, 2:],
    ])


def _spatial_min_excl_center(img: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise minimum of the 8 spatial neighbours, excluding centre."""
    return np.minimum.reduce([
        img[:-2, :-2], img[:-2, 1:-1], img[:-2, 2:],
        img[1:-1, :-2],                img[1:-1, 2:],
        img[2:, :-2],  img[2:, 1:-1],  img[2:, 2:],
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_keypoints(
    dog_pyramid: list[list[NDArray[np.float64]]],
) -> list[tuple[int, int, int, int]]:
    """Find candidate keypoints as 3-D extrema in the DoG pyramid.

    Args:
        dog_pyramid: Output of :func:`build_dog_pyramid`.

    Returns:
        List of (octave, scale, row, col) tuples for each candidate.
    """
    candidates: list[tuple[int, int, int, int]] = []

    for o, dog_octave in enumerate(dog_pyramid):
        num_dog = len(dog_octave)
        if num_dog < 3:
            continue

        for s in range(1, num_dog - 1):
            D_prev = dog_octave[s - 1]
            D_curr = dog_octave[s]
            D_next = dog_octave[s + 1]

            interior = D_curr[1:-1, 1:-1]

            # 26-neighbour max / min (vectorised)
            neighbor_max = np.maximum.reduce([
                _layer_max(D_prev),
                _spatial_max_excl_center(D_curr),
                _layer_max(D_next),
            ])
            neighbor_min = np.minimum.reduce([
                _layer_min(D_prev),
                _spatial_min_excl_center(D_curr),
                _layer_min(D_next),
            ])

            is_extremum = (interior > neighbor_max) | (interior < neighbor_min)
            rows, cols = np.where(is_extremum)

            for r, c in zip(rows + 1, cols + 1):   # +1: restore border offset
                candidates.append((o, s, int(r), int(c)))

    return candidates


def localize_keypoints(
    dog_pyramid: list[list[NDArray[np.float64]]],
    candidates: list[tuple[int, int, int, int]],
    contrast_threshold: float = CONTRAST_THRESHOLD,
    edge_threshold: float = EDGE_THRESHOLD,
) -> list[dict]:
    """Refine keypoint locations and discard low-quality candidates.

    Applies:
    - Taylor-series sub-pixel / sub-scale interpolation
    - Contrast threshold to remove low-contrast points
    - Hessian-based edge response test (Harris-like ratio)

    Args:
        dog_pyramid: Output of :func:`build_dog_pyramid`.
        candidates: Candidate keypoints from :func:`detect_keypoints`.
        contrast_threshold: Minimum |D(x̂)| to keep a keypoint.
        edge_threshold: Maximum principal-curvature ratio threshold.

    Returns:
        List of dicts with keys: octave, scale, row, col, sigma, response.
    """
    keypoints: list[dict] = []

    for (o, s0, r0, c0) in candidates:
        dog_octave = dog_pyramid[o]
        num_dog = len(dog_octave)
        num_scales = num_dog - 2
        h, w = dog_octave[0].shape

        s, r, c = s0, r0, c0
        converged = False
        grad = np.zeros(3)
        drr = drc = dcc = 0.0
        offset = np.zeros(3)

        for _ in range(_MAX_INTERP_ITER):
            if not (1 <= s < num_dog - 1 and 1 <= r < h - 1 and 1 <= c < w - 1):
                break

            val = dog_octave[s][r, c]

            # First derivatives (central differences)
            grad = np.array([
                (dog_octave[s + 1][r, c]     - dog_octave[s - 1][r, c])     * 0.5,
                (dog_octave[s][r + 1, c]     - dog_octave[s][r - 1, c])     * 0.5,
                (dog_octave[s][r, c + 1]     - dog_octave[s][r, c - 1])     * 0.5,
            ])

            # Second derivatives
            dss = dog_octave[s + 1][r, c]     - 2.0 * val + dog_octave[s - 1][r, c]
            drr = dog_octave[s][r + 1, c]     - 2.0 * val + dog_octave[s][r - 1, c]
            dcc = dog_octave[s][r, c + 1]     - 2.0 * val + dog_octave[s][r, c - 1]
            dsr = (dog_octave[s+1][r+1,c] - dog_octave[s+1][r-1,c]
                 - dog_octave[s-1][r+1,c] + dog_octave[s-1][r-1,c]) * 0.25
            dsc = (dog_octave[s+1][r,c+1] - dog_octave[s+1][r,c-1]
                 - dog_octave[s-1][r,c+1] + dog_octave[s-1][r,c-1]) * 0.25
            drc = (dog_octave[s][r+1,c+1] - dog_octave[s][r+1,c-1]
                 - dog_octave[s][r-1,c+1] + dog_octave[s][r-1,c-1]) * 0.25

            H = np.array([
                [dss, dsr, dsc],
                [dsr, drr, drc],
                [dsc, drc, dcc],
            ])

            try:
                offset = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                break

            if np.all(np.abs(offset) < 0.5):
                converged = True
                break

            s += int(np.round(offset[0]))
            r += int(np.round(offset[1]))
            c += int(np.round(offset[2]))

        if not converged:
            continue

        # --- Contrast check ---
        response = float(dog_octave[s][r, c] + 0.5 * (grad @ offset))
        if abs(response) < contrast_threshold:
            continue

        # --- Edge response check (2×2 spatial Hessian, Lowe 2004 §4.1) ---
        trace_H = drr + dcc
        det_H = drr * dcc - drc * drc
        if det_H <= 0.0 or trace_H ** 2 / det_H >= (edge_threshold + 1.0) ** 2 / edge_threshold:
            continue

        # --- Absolute sigma in original-image pixels ---
        # Divide by 2 because octave 0 is the 2× upsampled image (Lowe 2004 §3).
        k = 2.0 ** (1.0 / num_scales)
        sigma = _BASE_SIGMA * (k ** (s + float(offset[0]))) * (2.0 ** o) / 2.0

        keypoints.append({
            "octave":   o,
            "scale":    s,
            "row":      float(r) + float(offset[1]),
            "col":      float(c) + float(offset[2]),
            "sigma":    sigma,
            "response": response,
        })

    return keypoints

