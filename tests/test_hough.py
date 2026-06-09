"""Tests for Hough-voting geometric verification."""

import numpy as np
import pytest

from sift.hough import geometric_verification, _fit_affine, _affine_residuals


RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Helper: build synthetic keypoint dicts and matches under a known similarity
# ---------------------------------------------------------------------------

def _make_kp(x: float, y: float, sigma: float = 2.0, angle: float = 0.0,
             octave: int = 0) -> dict:
    """Create a minimal keypoint dict in image-plane coordinates.

    row/col are set so that _kp_xy returns (x, y) correctly:
        _kp_xy = (col * 2^oct / 2, row * 2^oct / 2)
    → col = x * 2 / 2^oct,  row = y * 2 / 2^oct
    """
    scale = 2.0 ** octave
    return {
        "col": x * 2.0 / scale,
        "row": y * 2.0 / scale,
        "sigma": sigma,
        "angle": angle,
        "octave": octave,
    }


def _make_synthetic_matches(
    n_inliers: int = 40,
    n_outliers: int = 10,
    noise_std: float = 0.3,
    rot_deg: float = 5.0,
    scale_ratio: float = 1.1,
    tx: float = 15.0,
    ty: float = -10.0,
    img_size: int = 500,
):
    """Generate (kps1, kps2, matches, gt_mask) under a known similarity."""
    rot = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot), np.sin(rot)

    rng = np.random.default_rng(42)
    xs = rng.uniform(50, img_size - 50, n_inliers)
    ys = rng.uniform(50, img_size - 50, n_inliers)

    sigma_base = 3.0
    kps1, kps2, matches = [], [], []

    for i, (x, y) in enumerate(zip(xs, ys)):
        kp1 = _make_kp(x, y, sigma=sigma_base, angle=0.1 * i % (2 * np.pi))
        # Apply similarity transform + noise
        xp = scale_ratio * (cos_r * x - sin_r * y) + tx + rng.normal(0, noise_std)
        yp = scale_ratio * (sin_r * x + cos_r * y) + ty + rng.normal(0, noise_std)
        angle2 = (kp1["angle"] + rot) % (2 * np.pi)
        kp2 = _make_kp(xp, yp, sigma=sigma_base * scale_ratio, angle=angle2)
        matches.append((i, i, 0.1))
        kps1.append(kp1)
        kps2.append(kp2)

    # Outliers: random correspondences
    for j in range(n_outliers):
        i1 = len(kps1)
        i2 = len(kps2)
        kps1.append(_make_kp(rng.uniform(50, img_size - 50),
                             rng.uniform(50, img_size - 50),
                             sigma=sigma_base, angle=rng.uniform(0, 2 * np.pi)))
        kps2.append(_make_kp(rng.uniform(50, img_size - 50),
                             rng.uniform(50, img_size - 50),
                             sigma=sigma_base, angle=rng.uniform(0, 2 * np.pi)))
        matches.append((i1, i2, 0.5))

    gt_mask = np.zeros(len(matches), dtype=bool)
    gt_mask[:n_inliers] = True

    return kps1, kps2, matches, gt_mask, (img_size, img_size)


# ---------------------------------------------------------------------------
# _fit_affine
# ---------------------------------------------------------------------------

class TestFitAffine:
    def test_identity(self):
        pts = RNG.uniform(0, 200, (10, 2))
        result = _fit_affine(pts, pts)
        assert result is not None
        M, t = result
        assert np.allclose(M, np.eye(2), atol=1e-6)
        assert np.allclose(t, 0.0, atol=1e-6)

    def test_known_transform(self):
        rot = np.deg2rad(20)
        M_true = np.array([[np.cos(rot), -np.sin(rot)],
                           [np.sin(rot),  np.cos(rot)]]) * 1.3
        t_true = np.array([10.0, -5.0])
        pts1 = RNG.uniform(0, 200, (15, 2))
        pts2 = (M_true @ pts1.T).T + t_true
        result = _fit_affine(pts1, pts2)
        assert result is not None
        M, t = result
        assert np.allclose(M, M_true, atol=1e-6)
        assert np.allclose(t, t_true, atol=1e-6)

    def test_too_few_points(self):
        pts = RNG.uniform(0, 100, (2, 2))
        assert _fit_affine(pts, pts) is None


# ---------------------------------------------------------------------------
# geometric_verification
# ---------------------------------------------------------------------------

class TestGeometricVerification:
    def test_identity_all_inliers(self):
        """Same keypoints should yield all inliers with identity affine."""
        kps1, kps2, matches, gt_mask, shape = _make_synthetic_matches(
            n_inliers=30, n_outliers=0
        )
        affine, mask = geometric_verification(kps1, kps2, matches, shape)
        assert affine is not None
        assert mask.sum() >= 25   # almost all inliers recovered

    def test_recovers_inliers(self):
        """Most ground-truth inliers should be recovered."""
        kps1, kps2, matches, gt_mask, shape = _make_synthetic_matches(
            n_inliers=40, n_outliers=10
        )
        affine, mask = geometric_verification(kps1, kps2, matches, shape)
        assert affine is not None
        recall = (mask & gt_mask).sum() / gt_mask.sum()
        assert recall >= 0.80, f"recall {recall:.2f} too low"

    def test_rejects_outliers(self):
        """Ground-truth outliers should mostly be rejected."""
        kps1, kps2, matches, gt_mask, shape = _make_synthetic_matches(
            n_inliers=40, n_outliers=10
        )
        _, mask = geometric_verification(kps1, kps2, matches, shape)
        n_out = int((~gt_mask).sum())
        if n_out > 0:
            fp_rate = (mask & ~gt_mask).sum() / n_out
            assert fp_rate <= 0.5, f"FP rate {fp_rate:.2f} too high"

    def test_too_few_matches(self):
        kps1, kps2, matches, _, shape = _make_synthetic_matches(n_inliers=2, n_outliers=0)
        affine, mask = geometric_verification(kps1, kps2, matches[:2], shape)
        assert affine is None
        assert not mask.any()

    def test_mask_shape(self):
        kps1, kps2, matches, _, shape = _make_synthetic_matches()
        _, mask = geometric_verification(kps1, kps2, matches, shape)
        assert mask.shape == (len(matches),)
        assert mask.dtype == bool

    def test_no_geometric_structure(self):
        """Fully random matches with no structure → None or very few inliers."""
        rng = np.random.default_rng(99)
        n = 20
        kps1 = [_make_kp(*rng.uniform(50, 450, 2), sigma=rng.uniform(1, 5),
                         angle=rng.uniform(0, 2 * np.pi)) for _ in range(n)]
        kps2 = [_make_kp(*rng.uniform(50, 450, 2), sigma=rng.uniform(1, 5),
                         angle=rng.uniform(0, 2 * np.pi)) for _ in range(n)]
        matches = [(i, i, 0.5) for i in range(n)]
        _, mask = geometric_verification(kps1, kps2, matches, (500, 500),
                                         affine_thresh=3.0)
        # With a tight threshold, random data should yield very few inliers
        assert mask.sum() < n // 2

