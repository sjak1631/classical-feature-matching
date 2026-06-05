"""
Scale space construction.

Implements:
- Gaussian pyramid (octaves × scales)
- Difference-of-Gaussian (DoG) pyramid
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_gaussian_pyramid(
    image: NDArray[np.float64],
    num_octaves: int,
    num_scales: int,
    sigma: float = 1.6,
) -> list[list[NDArray[np.float64]]]:
    """Build a Gaussian scale-space pyramid.

    Args:
        image: Grayscale input image, float64 in [0, 1].
        num_octaves: Number of octaves.
        num_scales: Number of scale levels per octave (s in Lowe 2004).
        sigma: Base sigma for the first scale level.

    Returns:
        pyramid[octave][scale] = blurred image.
    """
    raise NotImplementedError


def build_dog_pyramid(
    gaussian_pyramid: list[list[NDArray[np.float64]]],
) -> list[list[NDArray[np.float64]]]:
    """Build the Difference-of-Gaussian pyramid from a Gaussian pyramid.

    Args:
        gaussian_pyramid: Output of :func:`build_gaussian_pyramid`.

    Returns:
        dog[octave][scale] = DoG image  (len = num_scales + 2 per octave).
    """
    raise NotImplementedError
