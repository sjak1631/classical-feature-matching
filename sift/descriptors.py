"""
SIFT descriptor computation.

Implements:
- 4×4 spatial grid of 8-bin gradient histograms → 128-D descriptor
- Gaussian weighting of gradient magnitudes
- Descriptor normalisation and clamping (Lowe 2004, §6.1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


DESCRIPTOR_SIZE = 4   # 4×4 spatial grid
NUM_ORIENT_BINS = 8   # 8 orientation bins per cell


def compute_descriptors(
    gaussian_pyramid: list[list[NDArray[np.float64]]],
    keypoints: list[dict],
) -> NDArray[np.float64]:
    """Compute a 128-D SIFT descriptor for each keypoint.

    Args:
        gaussian_pyramid: Output of :func:`build_gaussian_pyramid`.
        keypoints: Oriented keypoints from :func:`assign_orientations`.

    Returns:
        Array of shape (N, 128) with float64 descriptors, each L2-normalised.
    """
    raise NotImplementedError
