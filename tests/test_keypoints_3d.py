"""
Tests for keypoints/keypoints_3d.py

Covers: principal-point projection, off-axis projection,
invalid depth handling, to_3d / to_2d roundtrip.
"""
import numpy as np
import pytest
from modules.keypoints.keypoints_3d import Keypoints3DXform
from tests.conftest import MockCameraParams


@pytest.fixture
def xform(K_standard, camera_params):
    return Keypoints3DXform(camera_params)


# ---------------------------------------------------------------------------
# to_3d
# ---------------------------------------------------------------------------

class TestTo3D:

    def test_principal_point_maps_to_zero_xy(self, xform):
        """Point at the principal point (cx, cy) must project to (0, 0, Z)."""
        depth = np.full((480, 640), 5.0)
        pts = np.array([[320.0, 240.0]])  # cx, cy
        result = xform.to_3d(pts, depth)
        assert np.allclose(result[0], [0.0, 0.0, 5.0], atol=1e-9)

    def test_off_axis_point(self, xform, K_standard):
        """Off-axis point: X = (u - cx)/fx * Z."""
        Z = 3.0
        u, v = 420.0, 290.0
        depth = np.full((480, 640), Z)
        pts = np.array([[u, v]])
        result = xform.to_3d(pts, depth)
        fx, fy = K_standard[0, 0], K_standard[1, 1]
        cx, cy = K_standard[0, 2], K_standard[1, 2]
        assert np.allclose(result[0, 0], (u - cx) / fx * Z, atol=1e-9)
        assert np.allclose(result[0, 1], (v - cy) / fy * Z, atol=1e-9)
        assert np.allclose(result[0, 2], Z, atol=1e-9)

    def test_zero_depth_gives_zero_point(self, xform):
        """Zero depth is invalid — must return [0, 0, 0]."""
        depth = np.zeros((480, 640))
        pts = np.array([[100.0, 100.0]])
        result = xform.to_3d(pts, depth)
        assert np.allclose(result[0], [0.0, 0.0, 0.0])

    def test_negative_depth_gives_zero_point(self, xform):
        """Negative depth is invalid — must return [0, 0, 0]."""
        depth = np.full((480, 640), -1.0)
        pts = np.array([[100.0, 100.0]])
        result = xform.to_3d(pts, depth)
        assert np.allclose(result[0], [0.0, 0.0, 0.0])

    def test_multiple_points(self, xform):
        """Batch of points all get correct Z values."""
        Z = 4.0
        depth = np.full((480, 640), Z)
        pts = np.array([[320.0, 240.0], [100.0, 50.0], [500.0, 400.0]])
        result = xform.to_3d(pts, depth)
        assert result.shape == (3, 3)
        assert np.allclose(result[:, 2], Z)

    def test_depth_map_2d_required(self, xform):
        """Passing a 3D array for depth_map must raise."""
        pts = np.array([[100.0, 100.0]])
        bad_depth = np.ones((480, 640, 1))
        with pytest.raises(ValueError, match="2D"):
            xform.to_3d(pts, bad_depth)

    def test_mixed_valid_invalid_depths(self, xform):
        """Only valid-depth points are non-zero; others are [0,0,0]."""
        depth = np.zeros((480, 640))
        depth[100, 200] = 2.0   # valid only here
        pts = np.array([[200.0, 100.0], [300.0, 200.0]])
        result = xform.to_3d(pts, depth)
        assert result[0, 2] == pytest.approx(2.0)
        assert np.allclose(result[1], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# to_2d
# ---------------------------------------------------------------------------

class TestTo2D:

    def test_principal_axis_point(self, xform):
        """(0, 0, Z) must project back to the principal point."""
        pts_3d = np.array([[0.0, 0.0, 5.0]])
        result = xform.to_2d(pts_3d)
        assert np.allclose(result[0], [320.0, 240.0], atol=1e-6)

    def test_skips_nonpositive_z(self, xform):
        """Points with Z <= 0 must be skipped in the output."""
        pts_3d = np.array([[1.0, 1.0, 3.0], [0.0, 0.0, 0.0], [1.0, 1.0, -1.0]])
        result = xform.to_2d(pts_3d)
        assert result.shape == (1, 2)   # only the first point is valid


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:

    def test_to3d_then_to2d(self, xform):
        """to_3d followed by to_2d must recover the original pixel coordinates."""
        Z = 5.0
        depth = np.full((480, 640), Z)
        pts_2d = np.array([[100.0, 150.0], [320.0, 240.0], [500.0, 400.0]])
        pts_3d = xform.to_3d(pts_2d, depth)
        pts_2d_back = xform.to_2d(pts_3d)
        assert np.allclose(pts_2d_back, pts_2d, atol=1e-6)

    def test_roundtrip_varying_depths(self, xform):
        """Roundtrip must work even when each pixel has a different depth."""
        depth = np.ones((480, 640))
        # Set specific depths for our test pixels
        coords = [(100, 200, 1.5), (300, 150, 3.0), (450, 350, 8.0)]  # (u, v, Z)
        pts_2d = np.array([[u, v] for u, v, _ in coords], dtype=float)
        for u, v, z in coords:
            depth[v, u] = z

        pts_3d = xform.to_3d(pts_2d, depth)
        pts_2d_back = xform.to_2d(pts_3d)
        assert np.allclose(pts_2d_back, pts_2d, atol=1e-6)
