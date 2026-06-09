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

_CLAMP_THRESH = 0.2   # Lowe (2004) §6.1


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
    D = DESCRIPTOR_SIZE   # 4
    O = NUM_ORIENT_BINS   # 8
    desc_dim = D * D * O  # 128

    if not keypoints:
        return np.zeros((0, desc_dim), dtype=np.float64)

    descriptors = np.zeros((len(keypoints), desc_dim), dtype=np.float64)

    for i, kp in enumerate(keypoints):
        o_idx = kp["octave"]
        s_idx = kp["scale"]
        img = gaussian_pyramid[o_idx][s_idx]
        h, w = img.shape

        r = kp["row"]
        c = kp["col"]
        angle = kp.get("angle", 0.0)
        sigma_kp = kp["sigma"] * 2.0 / (2.0 ** o_idx)   # scale in octave pixels (accounts for 2× upsample)

        # Each of the 4 cells spans `cell_size` pixels.
        # Lowe uses λ = 3, so cell_size = 3σ.
        cell_size = 3.0 * sigma_kp
        half_win = D / 2.0 * cell_size   # half-window in pixels

        # Bounding box — enlarge by √2 to cover the rotated window
        radius = int(np.round(half_win * np.sqrt(2))) + 1
        r0 = max(int(r) - radius, 1)
        r1 = min(int(r) + radius + 1, h - 1)
        c0 = max(int(c) - radius, 1)
        c1 = min(int(c) + radius + 1, w - 1)

        if r1 <= r0 or c1 <= c0:
            continue

        # Central-difference gradients
        patch = img[r0 - 1 : r1 + 1, c0 - 1 : c1 + 1]
        dy = (patch[2:, 1:-1] - patch[:-2, 1:-1]) * 0.5
        dx = (patch[1:-1, 2:] - patch[1:-1, :-2]) * 0.5
        mag = np.sqrt(dx ** 2 + dy ** 2)
        ori = np.arctan2(dy, dx)          # [-π, π]

        # Relative pixel positions from keypoint centre
        rows_off = np.arange(r0, r1, dtype=np.float64) - r
        cols_off = np.arange(c0, c1, dtype=np.float64) - c
        dr, dc = np.meshgrid(rows_off, cols_off, indexing="ij")

        # Rotate to keypoint's canonical frame (undo dominant orientation)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dc_rot =  cos_a * dc + sin_a * dr
        dr_rot = -sin_a * dc + cos_a * dr

        # Continuous cell-grid coordinates in [-0.5, D-0.5]
        # (0.5 = centre of cell 0, 1.5 = centre of cell 1, …)
        cell_c = dc_rot / cell_size + D / 2.0 - 0.5
        cell_r = dr_rot / cell_size + D / 2.0 - 0.5

        # Gaussian spatial weight (σ = half the grid in cell units)
        gauss_sigma_px = (D / 2.0) * cell_size
        gauss_w = np.exp(-(dr_rot ** 2 + dc_rot ** 2) / (2.0 * gauss_sigma_px ** 2))

        # Relative gradient orientation, mapped to [0, O)
        rel_ori = (ori - angle) % (2.0 * np.pi)
        bin_f = rel_ori / (2.0 * np.pi) * O        # continuous bin index

        w_mag = mag * gauss_w

        # Mask to samples inside the descriptor window
        valid = (cell_r > -1.0) & (cell_r < D) & (cell_c > -1.0) & (cell_c < D)
        v_cr = cell_r[valid]
        v_cc = cell_c[valid]
        v_bf = bin_f[valid]
        v_wm = w_mag[valid]

        desc = np.zeros(desc_dim, dtype=np.float64)

        # Trilinear interpolation over (row_cell, col_cell, orientation_bin)
        ir0 = np.floor(v_cr).astype(int)
        ic0 = np.floor(v_cc).astype(int)
        ib0 = np.floor(v_bf).astype(int)

        fr = v_cr - ir0
        fc = v_cc - ic0
        fb = v_bf - ib0

        for dr_i in range(2):
            wr = (1.0 - fr) if dr_i == 0 else fr
            row_bin = ir0 + dr_i
            mask_r = (row_bin >= 0) & (row_bin < D)
            for dc_i in range(2):
                wc = (1.0 - fc) if dc_i == 0 else fc
                col_bin = ic0 + dc_i
                mask_c = (col_bin >= 0) & (col_bin < D)
                for db_i in range(2):
                    wb = (1.0 - fb) if db_i == 0 else fb
                    ori_bin = (ib0 + db_i) % O
                    mask = mask_r & mask_c
                    flat_idx = (row_bin * D + col_bin) * O + ori_bin
                    np.add.at(desc, flat_idx[mask], (wr * wc * wb * v_wm)[mask])

        # L2 normalise → clamp → renormalise  (Lowe 2004, §6.1)
        norm = np.linalg.norm(desc)
        if norm > 1e-12:
            desc /= norm
        np.clip(desc, 0.0, _CLAMP_THRESH, out=desc)
        norm2 = np.linalg.norm(desc)
        if norm2 > 1e-12:
            desc /= norm2

        descriptors[i] = desc

    return descriptors

