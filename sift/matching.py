"""
Feature matching.

Implements:
- Brute-force nearest-neighbour matching (L2 distance)
- Lowe's ratio test to filter ambiguous matches
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


RATIO_THRESHOLD = 0.75  # Lowe (2004) recommended value


def match_descriptors(
    desc1: NDArray[np.float64],
    desc2: NDArray[np.float64],
    ratio_threshold: float = RATIO_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Match descriptors using brute-force NN search + ratio test.

    Args:
        desc1: Descriptors from image 1, shape (N, 128).
        desc2: Descriptors from image 2, shape (M, 128).
        ratio_threshold: Lowe ratio test threshold.

    Returns:
        List of (idx1, idx2, distance) for passing matches.
    """
    raise NotImplementedError
