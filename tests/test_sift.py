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


def test_build_gaussian_pyramid_raises(sample_image):
    with pytest.raises(NotImplementedError):
        build_gaussian_pyramid(sample_image, num_octaves=3, num_scales=3)


def test_build_dog_pyramid_raises():
    with pytest.raises(NotImplementedError):
        build_dog_pyramid([[]])


def test_detect_keypoints_raises():
    with pytest.raises(NotImplementedError):
        detect_keypoints([[]])


def test_localize_keypoints_raises():
    with pytest.raises(NotImplementedError):
        localize_keypoints([[]], [])


def test_assign_orientations_raises():
    with pytest.raises(NotImplementedError):
        assign_orientations([[]], [])


def test_compute_descriptors_raises():
    with pytest.raises(NotImplementedError):
        compute_descriptors([[]], [])


def test_match_descriptors_raises():
    with pytest.raises(NotImplementedError):
        match_descriptors(np.zeros((1, 128)), np.zeros((1, 128)))
