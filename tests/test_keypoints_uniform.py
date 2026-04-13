"""
Tests for keypoints/keypoints_uniform.py

Covers: count, mask filtering, margin, output shape/dtype.
"""
import numpy as np
import pytest
from keypoints.keypoints_uniform import UniformKeyPoints


@pytest.fixture
def full_mask():
    return np.ones((480, 640), dtype=bool)


@pytest.fixture
def half_mask():
    """Left half of the image is valid, right half is invalid."""
    mask = np.zeros((480, 640), dtype=bool)
    mask[:, :320] = True
    return mask


class TestUniformKeyPoints:

    def test_output_shape(self, full_mask):
        kp = UniformKeyPoints(full_mask)
        pts = kp.get_keypoints(None, max_number=100)
        assert pts.ndim == 2
        assert pts.shape[1] == 2

    def test_respects_max_number(self, full_mask):
        kp = UniformKeyPoints(full_mask)
        pts = kp.get_keypoints(None, max_number=50)
        assert len(pts) <= 50

    def test_all_points_inside_image(self, full_mask):
        h, w = full_mask.shape
        kp = UniformKeyPoints(full_mask)
        pts = kp.get_keypoints(None, max_number=200)
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < w)
        assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < h)

    def test_all_points_in_valid_mask(self, half_mask):
        """Points must only appear where the mask is True."""
        kp = UniformKeyPoints(half_mask)
        pts = kp.get_keypoints(None, max_number=200)
        assert len(pts) > 0
        # All u-coordinates must be in the valid left half (u < 320)
        assert np.all(pts[:, 0] < 320)

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((480, 640), dtype=bool)
        kp = UniformKeyPoints(mask)
        pts = kp.get_keypoints(None, max_number=100)
        assert len(pts) == 0

    def test_margin_excludes_border(self, full_mask):
        margin = 20
        kp = UniformKeyPoints(full_mask, margin=margin)
        pts = kp.get_keypoints(None, max_number=200)
        h, w = full_mask.shape
        assert np.all(pts[:, 0] >= margin) and np.all(pts[:, 0] < w - margin)
        assert np.all(pts[:, 1] >= margin) and np.all(pts[:, 1] < h - margin)

    def test_returns_integer_coordinates(self, full_mask):
        """Pixel coordinates must be integer-valued (stored as int or castable)."""
        kp = UniformKeyPoints(full_mask)
        pts = kp.get_keypoints(None, max_number=100)
        assert np.issubdtype(pts.dtype, np.integer)
