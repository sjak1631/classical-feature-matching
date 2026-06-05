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
CONTRAST_THRESHOLD = 0.04
EDGE_THRESHOLD = 10.0


def detect_keypoints(
    dog_pyramid: list[list[NDArray[np.float64]]],
) -> list[tuple[int, int, int, int]]:
    """Find candidate keypoints as 3-D extrema in the DoG pyramid.

    Args:
        dog_pyramid: Output of :func:`build_dog_pyramid`.

    Returns:
        List of (octave, scale, row, col) tuples for each candidate.
    """
    raise NotImplementedError


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
    raise NotImplementedError
