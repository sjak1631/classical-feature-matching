"""
Orientation assignment.

Implements:
- Gradient magnitude / phase computation
- Orientation histogram (36 bins) with Gaussian weighting
- Peak selection and parabolic interpolation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


NUM_BINS = 36
PEAK_RATIO = 0.8  # Secondary peaks above 80 % of main peak get their own keypoint

_BIN_WIDTH = 2.0 * np.pi / NUM_BINS        # radians per bin
_WIN_FACTOR = 1.5                           # Gaussian window σ = WIN_FACTOR * kp.sigma


def _smooth_histogram(hist: NDArray[np.float64], n_iter: int = 6) -> NDArray[np.float64]:
    """Circular Gaussian-like smoothing of the histogram (box-filter passes)."""
    h = hist.copy()
    for _ in range(n_iter):
        h = (np.roll(h, -1) + h + np.roll(h, 1)) / 3.0
    return h


def assign_orientations(
    gaussian_pyramid: list[list[NDArray[np.float64]]],
    keypoints: list[dict],
) -> list[dict]:
    """Assign one or more dominant orientations to each keypoint.

    Each keypoint may produce multiple entries (one per dominant orientation).

    Args:
        gaussian_pyramid: Output of :func:`build_gaussian_pyramid`.
        keypoints: Localized keypoints from :func:`localize_keypoints`.

    Returns:
        Keypoints with an added ``angle`` field (radians, 0–2π).
    """
    oriented: list[dict] = []

    for kp in keypoints:
        o = kp["octave"]
        s = kp["scale"]
        img = gaussian_pyramid[o][s]
        h, w = img.shape

        # Keypoint position in this octave's pixel grid
        r = kp["row"]
        c = kp["col"]
        sigma_kp = kp["sigma"] * 2.0 / (2.0 ** o)    # scale in octave pixels (accounts for 2× upsample)

        # Gaussian weighting window radius (3σ captures >99 % of the weight)
        win_sigma = _WIN_FACTOR * sigma_kp
        radius = int(np.round(3.0 * win_sigma))

        # Row / col integer bounds clamped to image interior (need ±1 for gradient)
        r0 = max(int(r) - radius, 1)
        r1 = min(int(r) + radius + 1, h - 1)
        c0 = max(int(c) - radius, 1)
        c1 = min(int(c) + radius + 1, w - 1)

        if r1 <= r0 or c1 <= c0:
            continue

        # Central-difference gradients over the window
        patch = img[r0 - 1 : r1 + 1, c0 - 1 : c1 + 1]
        dy = (patch[2:, 1:-1] - patch[:-2, 1:-1]) * 0.5
        dx = (patch[1:-1, 2:] - patch[1:-1, :-2]) * 0.5
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        orientation = np.arctan2(dy, dx) % (2.0 * np.pi)   # [0, 2π)

        # Gaussian spatial weighting
        rows_idx = np.arange(r0, r1, dtype=np.float64) - r
        cols_idx = np.arange(c0, c1, dtype=np.float64) - c
        rr, cc = np.meshgrid(rows_idx, cols_idx, indexing="ij")
        weight = np.exp(-(rr ** 2 + cc ** 2) / (2.0 * win_sigma ** 2))

        weighted_mag = magnitude * weight

        # Accumulate orientation histogram
        bin_idx = (orientation / _BIN_WIDTH).astype(int) % NUM_BINS
        hist = np.zeros(NUM_BINS, dtype=np.float64)
        np.add.at(hist, bin_idx, weighted_mag)

        hist = _smooth_histogram(hist)

        # Peak selection
        peak_val = hist.max()
        threshold = PEAK_RATIO * peak_val

        for b in range(NUM_BINS):
            if hist[b] < threshold:
                continue
            prev_b = (b - 1) % NUM_BINS
            next_b = (b + 1) % NUM_BINS
            # Only local maxima
            if hist[b] <= hist[prev_b] or hist[b] <= hist[next_b]:
                continue

            # Parabolic interpolation for sub-bin precision
            denom = hist[prev_b] - 2.0 * hist[b] + hist[next_b]
            if abs(denom) < 1e-12:
                frac = 0.0
            else:
                frac = 0.5 * (hist[prev_b] - hist[next_b]) / denom

            angle = ((b + 0.5 + frac) * _BIN_WIDTH) % (2.0 * np.pi)

            new_kp = dict(kp)
            new_kp["angle"] = float(angle)
            oriented.append(new_kp)

    return oriented

