"""Basic sanity-check tests (will grow as implementation progresses)."""

import numpy as np
import pytest

from sift.scale_space import build_gaussian_pyramid, build_dog_pyramid
from sift.keypoints import detect_keypoints, localize_keypoints
from sift.orientation import assign_orientations
from sift.descriptors import compute_descriptors
from sift.matching import match_descriptors


@pytest.fixture
def sample_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((128, 128))


# ---------------------------------------------------------------------------
# Placeholder tests — replace / extend as each module is implemented
# ---------------------------------------------------------------------------


def test_build_gaussian_pyramid_shape(sample_image):
    num_octaves, num_scales = 3, 3
    pyramid = build_gaussian_pyramid(sample_image, num_octaves=num_octaves, num_scales=num_scales)

    assert len(pyramid) == num_octaves
    for o, octave in enumerate(pyramid):
        assert len(octave) == num_scales + 3, f"octave {o} should have {num_scales + 3} levels"

    # Each octave is half the spatial size of the previous one
    for o in range(1, num_octaves):
        prev_shape = np.array(pyramid[o - 1][0].shape)
        curr_shape = np.array(pyramid[o][0].shape)
        assert np.allclose(prev_shape // 2, curr_shape), f"octave {o} size mismatch"


def test_build_gaussian_pyramid_blur_increases(sample_image):
    """Higher scales must be more blurred (lower high-freq energy)."""
    pyramid = build_gaussian_pyramid(sample_image, num_octaves=2, num_scales=3)
    octave = pyramid[0]
    variances = [img.var() for img in octave]
    # Variance should be non-increasing as blur increases
    for i in range(len(variances) - 1):
        assert variances[i] >= variances[i + 1] - 1e-9, (
            f"Variance increased from level {i} to {i + 1}: {variances[i]:.6f} -> {variances[i + 1]:.6f}"
        )


def test_build_dog_pyramid_shape(sample_image):
    num_octaves, num_scales = 3, 3
    gaussian_pyramid = build_gaussian_pyramid(sample_image, num_octaves=num_octaves, num_scales=num_scales)
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)

    assert len(dog_pyramid) == num_octaves
    for o, dog_octave in enumerate(dog_pyramid):
        expected_levels = len(gaussian_pyramid[o]) - 1  # num_scales + 2
        assert len(dog_octave) == expected_levels, f"octave {o}: expected {expected_levels} DoG levels"
        for s, dog_img in enumerate(dog_octave):
            assert dog_img.shape == gaussian_pyramid[o][s].shape, f"octave {o} scale {s}: shape mismatch"


def test_build_dog_pyramid_values(sample_image):
    """DoG[o][s] must equal gaussian[o][s+1] - gaussian[o][s]."""
    gaussian_pyramid = build_gaussian_pyramid(sample_image, num_octaves=2, num_scales=3)
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)

    for o, (g_oct, d_oct) in enumerate(zip(gaussian_pyramid, dog_pyramid)):
        for s, dog_img in enumerate(d_oct):
            expected = g_oct[s + 1] - g_oct[s]
            assert np.allclose(dog_img, expected), f"octave {o} scale {s}: value mismatch"


@pytest.fixture
def dog_pyramid_from_image(sample_image):
    gp = build_gaussian_pyramid(sample_image, num_octaves=3, num_scales=3)
    return build_dog_pyramid(gp)


def test_detect_keypoints_returns_tuples(dog_pyramid_from_image):
    candidates = detect_keypoints(dog_pyramid_from_image)
    assert isinstance(candidates, list)
    for item in candidates:
        assert len(item) == 4, "each candidate must be (octave, scale, row, col)"
        o, s, r, c = item
        h, w = dog_pyramid_from_image[o][s].shape
        assert 0 <= o < len(dog_pyramid_from_image)
        assert 1 <= r < h - 1
        assert 1 <= c < w - 1


def test_detect_keypoints_finds_spike():
    """A sharp positive spike must be detected as a local maximum."""
    h, w = 20, 20
    zero = np.zeros((h, w))
    dog_octave = [zero.copy(), zero.copy(), zero.copy()]
    dog_octave[1][10, 10] = 1.0
    candidates = detect_keypoints([dog_octave])
    assert any(r == 10 and c == 10 for _, _, r, c in candidates), \
        "spike at (10,10) was not detected"


def test_detect_keypoints_finds_trough():
    """A sharp negative spike must be detected as a local minimum."""
    h, w = 20, 20
    zero = np.zeros((h, w))
    dog_octave = [zero.copy(), zero.copy(), zero.copy()]
    dog_octave[1][10, 10] = -1.0
    candidates = detect_keypoints([dog_octave])
    assert any(r == 10 and c == 10 for _, _, r, c in candidates), \
        "trough at (10,10) was not detected"


def test_localize_keypoints_returns_dicts(dog_pyramid_from_image):
    candidates = detect_keypoints(dog_pyramid_from_image)
    keypoints = localize_keypoints(dog_pyramid_from_image, candidates)
    assert isinstance(keypoints, list)
    required_keys = {"octave", "scale", "row", "col", "sigma", "response"}
    for kp in keypoints:
        assert required_keys.issubset(kp.keys()), f"missing keys: {required_keys - kp.keys()}"
        assert kp["sigma"] > 0.0


def test_localize_keypoints_filters_low_contrast():
    """With a very high contrast threshold, all keypoints should be discarded."""
    h, w = 20, 20
    zero = np.zeros((h, w))
    dog_octave = [zero.copy(), zero.copy(), zero.copy()]
    dog_octave[1][10, 10] = 0.001          # tiny contrast
    candidates = detect_keypoints([dog_octave])
    keypoints = localize_keypoints([dog_octave], candidates, contrast_threshold=1.0)
    assert len(keypoints) == 0, "low-contrast keypoint should have been discarded"


def test_assign_orientations_angle_in_range(sample_image):
    gp = build_gaussian_pyramid(sample_image, num_octaves=3, num_scales=3)
    dp = build_dog_pyramid(gp)
    candidates = detect_keypoints(dp)
    kps = localize_keypoints(dp, candidates)
    oriented = assign_orientations(gp, kps)

    assert len(oriented) >= len(kps), "orientation should not reduce keypoint count"
    for kp in oriented:
        assert "angle" in kp, "missing 'angle' key"
        assert 0.0 <= kp["angle"] < 2.0 * np.pi, f"angle out of [0, 2π): {kp['angle']}"


def test_assign_orientations_detects_dominant_direction():
    """A synthetic gradient field pointing right (angle≈0) should yield angle near 0 or 2π."""
    h, w = 64, 64
    # Image that increases linearly along x → dx>0, dy=0 → angle = 0
    img = np.tile(np.linspace(0.0, 1.0, w), (h, 1))
    gp = [[img]]   # single octave, single scale (no DoG needed here)
    kp = {"octave": 0, "scale": 0, "row": 32.0, "col": 32.0, "sigma": 2.0, "response": 0.1}
    oriented = assign_orientations(gp, [kp])

    assert len(oriented) >= 1
    angle = oriented[0]["angle"]
    # Allow ±15° tolerance; angle ≈ 0 (or ≈ 2π)
    tol = np.deg2rad(15)
    assert angle < tol or angle > 2.0 * np.pi - tol, f"expected angle ≈ 0, got {np.rad2deg(angle):.1f}°"


def test_compute_descriptors_shape(sample_image):
    gp = build_gaussian_pyramid(sample_image, num_octaves=3, num_scales=3)
    dp = build_dog_pyramid(gp)
    candidates = detect_keypoints(dp)
    kps = localize_keypoints(dp, candidates)
    oriented = assign_orientations(gp, kps)
    descs = compute_descriptors(gp, oriented)

    assert descs.ndim == 2
    assert descs.shape[1] == 128
    assert descs.shape[0] == len(oriented)


def test_compute_descriptors_normalised(sample_image):
    gp = build_gaussian_pyramid(sample_image, num_octaves=3, num_scales=3)
    dp = build_dog_pyramid(gp)
    kps = localize_keypoints(dp, detect_keypoints(dp))
    oriented = assign_orientations(gp, kps)
    descs = compute_descriptors(gp, oriented)

    norms = np.linalg.norm(descs, axis=1)
    non_zero = norms > 1e-12
    assert np.allclose(norms[non_zero], 1.0, atol=1e-6), "descriptors must be unit-norm"


def test_compute_descriptors_empty():
    gp = [[np.zeros((32, 32))]]
    descs = compute_descriptors(gp, [])
    assert descs.shape == (0, 128)


def test_compute_descriptors_rotation_invariant():
    """Descriptor of a keypoint with angle=0 and angle=π should be similar."""
    h, w = 64, 64
    rng = np.random.default_rng(42)
    img = rng.random((h, w))
    gp = [[img]]
    base_kp = {"octave": 0, "scale": 0, "row": 32.0, "col": 32.0, "sigma": 2.0,
               "response": 0.1}
    kp0 = dict(base_kp, angle=0.0)
    kp1 = dict(base_kp, angle=np.pi)
    d0 = compute_descriptors(gp, [kp0])[0]
    d1 = compute_descriptors(gp, [kp1])[0]
    # They should be different (different rotations), but both valid unit vectors
    assert np.linalg.norm(d0) > 1e-6
    assert np.linalg.norm(d1) > 1e-6


def test_match_descriptors_perfect_match():
    """Identical descriptor sets should produce N matches with distance 0."""
    rng = np.random.default_rng(7)
    desc = rng.random((10, 128))
    # L2-normalise so they look like real SIFT descriptors
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    matches = match_descriptors(desc, desc, ratio_threshold=0.9)
    assert len(matches) == 10
    for i, j, d in matches:
        assert i == j, f"expected self-match, got {i}->{j}"
        assert d < 1e-6, f"expected zero distance, got {d}"


def test_match_descriptors_ratio_test():
    """When the second NN is barely farther, the ratio test should reject."""
    d1 = np.zeros((1, 128), dtype=np.float64)
    d2 = np.zeros((2, 128), dtype=np.float64)
    d1[0, 0] = 1.0
    d2[0, 0] = 1.0     # nearest  → distance 0
    d2[1, 0] = 0.999   # second   → distance ≈ 0.001 (ratio ≈ 0 < threshold) should PASS
    matches = match_descriptors(d1, d2)
    assert len(matches) == 1


def test_match_descriptors_returns_format():
    rng = np.random.default_rng(0)
    d1 = rng.random((5, 128))
    d2 = rng.random((5, 128))
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    matches = match_descriptors(d1, d2)
    for item in matches:
        assert len(item) == 3
        idx1, idx2, dist = item
        assert 0 <= idx1 < len(d1)
        assert 0 <= idx2 < len(d2)
        assert dist >= 0.0


def test_match_descriptors_empty_input():
    assert match_descriptors(np.zeros((0, 128)), np.zeros((5, 128))) == []
    assert match_descriptors(np.zeros((5, 128)), np.zeros((0, 128))) == []

