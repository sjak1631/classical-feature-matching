"""
Scale space construction.

Implements:
- Gaussian pyramid (octaves × scales)
- Difference-of-Gaussian (DoG) pyramid
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _gaussian_kernel(sigma: float, truncate: float = 4.0) -> NDArray[np.float64]:
    """1-D Gaussian kernel, normalised to sum to 1."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _gaussian_blur(image: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Separable 2-D Gaussian blur with reflect padding (vectorised, numpy only).

    Replaces apply_along_axis with a sliding-window sum over the full image,
    which is orders of magnitude faster on large images.
    """
    if sigma == 0.0:
        return image.copy()
    kernel = _gaussian_kernel(sigma)
    radius = len(kernel) // 2
    h, w = image.shape

    # --- Row-wise convolution ---
    padded = np.pad(image, ((0, 0), (radius, radius)), mode="reflect")
    out = np.zeros((h, w), dtype=np.float64)
    for i, k in enumerate(kernel):
        out += k * padded[:, i : i + w]

    # --- Column-wise convolution ---
    padded2 = np.pad(out, ((radius, radius), (0, 0)), mode="reflect")
    result = np.zeros((h, w), dtype=np.float64)
    for i, k in enumerate(kernel):
        result += k * padded2[i : i + h, :]

    return result


def _upsample2x(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Bilinear 2× upsampling (linear interpolation, Lowe 2004 §3)."""
    h, w = image.shape
    y = np.arange(2 * h) * 0.5   # sample positions in original: [0, 0.5, 1, ...]
    x = np.arange(2 * w) * 0.5
    y0 = np.floor(y).astype(int).clip(0, h - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(x).astype(int).clip(0, w - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (y - y0)[:, np.newaxis]   # (2h, 1)
    fx = (x - x0)[np.newaxis, :]   # (1, 2w)
    return (
        (1.0 - fy) * (1.0 - fx) * image[np.ix_(y0, x0)]
        + (1.0 - fy) * fx * image[np.ix_(y0, x1)]
        + fy * (1.0 - fx) * image[np.ix_(y1, x0)]
        + fy * fx * image[np.ix_(y1, x1)]
    )


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
        Each octave contains ``num_scales + 3`` images.
    """
    # Number of blurred images per octave.
    # +3: one extra on each end so DoG has num_scales+2 levels,
    # and extrema can be found in the inner num_scales levels.
    levels = num_scales + 3
    k = 2.0 ** (1.0 / num_scales)  # scale factor between adjacent levels

    # Lowe (2004) §3: upsample 2× before building the pyramid.
    # Original camera blur σ_camera = 0.5 px becomes 1.0 px after 2× upsampling.
    # Apply additional blur to reach the target σ₀ = sigma.
    DOUBLED_SIGMA = 1.0  # 2 × σ_camera in upsampled pixel units
    sigma_init = np.sqrt(max(sigma ** 2 - DOUBLED_SIGMA ** 2, 0.0))
    base = _gaussian_blur(_upsample2x(image.astype(np.float64)), sigma_init)

    pyramid: list[list[NDArray[np.float64]]] = []

    for _ in range(num_octaves):
        octave: list[NDArray[np.float64]] = [base]
        current_sigma = sigma

        for s in range(1, levels):
            # Absolute sigma for level s: σ_s = sigma * k^s
            next_sigma = sigma * (k ** s)
            # Incremental blur needed: sqrt(σ_next² − σ_current²)
            sigma_inc = np.sqrt(next_sigma ** 2 - current_sigma ** 2)
            blurred = _gaussian_blur(octave[-1], sigma_inc)
            octave.append(blurred)
            current_sigma = next_sigma

        pyramid.append(octave)

        # Seed of the next octave: image at level num_scales has effective
        # sigma = sigma * 2, so downsampling by 2 restores sigma spacing.
        base = octave[num_scales][::2, ::2]

    return pyramid
def build_dog_pyramid(
    gaussian_pyramid: list[list[NDArray[np.float64]]],
) -> list[list[NDArray[np.float64]]]:
    """Build the Difference-of-Gaussian pyramid from a Gaussian pyramid.

    Args:
        gaussian_pyramid: Output of :func:`build_gaussian_pyramid`.

    Returns:
        dog[octave][scale] = DoG image  (len = num_scales + 2 per octave).
    """
    dog_pyramid: list[list[NDArray[np.float64]]] = []
    for octave in gaussian_pyramid:
        dog_octave = [octave[i + 1] - octave[i] for i in range(len(octave) - 1)]
        dog_pyramid.append(dog_octave)
    return dog_pyramid
