"""
SIFT (Scale-Invariant Feature Transform) - scratch implementation.

References:
    Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.
    International Journal of Computer Vision, 60(2), 91-110.
"""

from .scale_space import build_gaussian_pyramid, build_dog_pyramid
from .keypoints import detect_keypoints, localize_keypoints
from .orientation import assign_orientations
from .descriptors import compute_descriptors
from .matching import match_descriptors
from .hough import geometric_verification

__all__ = [
    "build_gaussian_pyramid",
    "build_dog_pyramid",
    "detect_keypoints",
    "localize_keypoints",
    "assign_orientations",
    "compute_descriptors",
    "match_descriptors",
    "geometric_verification",
]
