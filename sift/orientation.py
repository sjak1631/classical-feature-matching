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
    raise NotImplementedError
