"""
Tests for camera/camera_svd_xform.py

Covers: Kabsch algorithm correctness, RANSAC outlier rejection,
edge cases (identity, reflection, minimal point sets).
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from modules.pose.camera_svd_xform import CameraSvdXform, CameraRansacXform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rotation(axis: str, degrees: float) -> np.ndarray:
    return Rotation.from_euler(axis, degrees, degrees=True).as_matrix()


def apply_xform(P: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ P.T).T + t


# A non-degenerate point cloud (not collinear, not coplanar)
POINTS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [2.0, 3.0, 4.0],
])


# ---------------------------------------------------------------------------
# CameraSvdXform — Kabsch
# ---------------------------------------------------------------------------

class TestCameraSvdXform:

    def test_identity_motion(self):
        """No motion: R=I, t=0."""
        xform = CameraSvdXform()
        R, t = xform.compute_camera_xform(POINTS, POINTS.copy())
        assert np.allclose(R, np.eye(3), atol=1e-9)
        assert np.allclose(t, np.zeros(3), atol=1e-9)

    def test_pure_translation(self):
        t_true = np.array([1.0, 2.0, 3.0])
        Q = POINTS + t_true
        xform = CameraSvdXform()
        R, t = xform.compute_camera_xform(POINTS, Q)
        assert np.allclose(R, np.eye(3), atol=1e-9)
        assert np.allclose(t, t_true, atol=1e-9)

    def test_pure_rotation_x(self):
        R_true = make_rotation('x', 30)
        Q = apply_xform(POINTS, R_true, np.zeros(3))
        xform = CameraSvdXform()
        R, t = xform.compute_camera_xform(POINTS, Q)
        assert np.allclose(R, R_true, atol=1e-9)
        assert np.allclose(t, np.zeros(3), atol=1e-9)

    def test_pure_rotation_y(self):
        R_true = make_rotation('y', 45)
        Q = apply_xform(POINTS, R_true, np.zeros(3))
        xform = CameraSvdXform()
        R, t = xform.compute_camera_xform(POINTS, Q)
        assert np.allclose(R, R_true, atol=1e-9)
        assert np.allclose(t, np.zeros(3), atol=1e-9)

    def test_combined_rotation_and_translation(self):
        R_true = make_rotation('z', 60)
        t_true = np.array([0.5, -1.0, 2.0])
        Q = apply_xform(POINTS, R_true, t_true)
        xform = CameraSvdXform()
        R, t = xform.compute_camera_xform(POINTS, Q)
        assert np.allclose(R, R_true, atol=1e-9)
        assert np.allclose(t, t_true, atol=1e-9)

    def test_rotation_matrix_is_proper(self):
        """Result must be a proper rotation: det(R)=+1 and R @ R^T = I."""
        R_true = make_rotation('xyz', [37, 20, 15])
        t_true = np.array([1.0, 0.0, -1.0])
        Q = apply_xform(POINTS, R_true, t_true)
        xform = CameraSvdXform()
        R, _ = xform.compute_camera_xform(POINTS, Q)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)

    def test_minimum_three_points(self):
        """Kabsch must work with exactly 3 non-collinear points."""
        P = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        t_true = np.array([1.0, 1.0, 1.0])
        Q = P + t_true
        xform = CameraSvdXform()
        R, t = xform.compute_camera_xform(P, Q)
        assert np.allclose(R, np.eye(3), atol=1e-9)
        assert np.allclose(t, t_true, atol=1e-9)

    def test_offset_applied_to_translation(self):
        """Camera offset must be folded into the returned translation."""
        offset = np.array([0.05, 0.0, 0.01])
        t_true = np.array([1.0, 0.0, 0.0])
        Q = POINTS + t_true
        xform = CameraSvdXform(offset=offset)
        R, t = xform.compute_camera_xform(POINTS, Q)
        # With R≈I: t_corrected = t + R @ offset ≈ t_true + offset
        assert np.allclose(t, t_true + offset, atol=1e-9)

    def test_large_rotation_no_reflection(self):
        """180-degree rotations must not produce reflections."""
        R_true = make_rotation('z', 180)
        Q = apply_xform(POINTS, R_true, np.zeros(3))
        xform = CameraSvdXform()
        R, _ = xform.compute_camera_xform(POINTS, Q)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        assert np.allclose(R, R_true, atol=1e-9)


# ---------------------------------------------------------------------------
# CameraRansacXform
# ---------------------------------------------------------------------------

class TestCameraRansacXform:

    def test_clean_data_matches_kabsch(self):
        """Without outliers RANSAC should agree with Kabsch."""
        R_true = make_rotation('y', 20)
        t_true = np.array([0.3, 0.1, -0.2])
        Q = apply_xform(POINTS, R_true, t_true)
        xform = CameraRansacXform(threshold=0.01, iterations=500)
        R, t = xform.compute_camera_xform(POINTS, Q)
        assert np.allclose(R, R_true, atol=1e-6)
        assert np.allclose(t, t_true, atol=1e-6)

    def test_outlier_rejection(self):
        """Gross outliers must not corrupt the result."""
        np.random.seed(42)
        P = np.random.randn(30, 3)
        R_true = make_rotation('z', 25)
        t_true = np.array([1.0, -0.5, 0.2])
        Q = apply_xform(P, R_true, t_true)

        # Corrupt 5 of 30 points with large errors
        Q_noisy = Q.copy()
        outlier_idx = [0, 5, 10, 15, 20]
        Q_noisy[outlier_idx] += np.random.uniform(5, 10, (5, 3))

        xform = CameraRansacXform(threshold=0.05, iterations=1000)
        R, t = xform.compute_camera_xform(P, Q_noisy)
        assert np.allclose(R, R_true, atol=0.01)
        assert np.allclose(t, t_true, atol=0.01)

    def test_too_few_points_raises(self):
        P = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        Q = P.copy()
        xform = CameraRansacXform()
        with pytest.raises(ValueError, match="3 points"):
            xform.compute_camera_xform(P, Q)

    def test_mismatched_shapes_raises(self):
        P = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        Q = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        xform = CameraRansacXform()
        with pytest.raises(ValueError):
            xform.compute_camera_xform(P, Q)

    def test_fallback_on_all_outliers(self):
        """When no inlier consensus exists, falls back to I, 0."""
        np.random.seed(0)
        P = np.random.randn(10, 3)
        Q = np.random.randn(10, 3) * 100  # completely unrelated
        xform = CameraRansacXform(threshold=0.001, iterations=100)
        R, t = xform.compute_camera_xform(P, Q)
        # fallback: identity rotation and zero translation
        assert np.allclose(R, np.eye(3))
        assert np.allclose(t, np.zeros(3))

    def test_result_is_proper_rotation(self):
        """RANSAC result must always be a proper rotation matrix."""
        np.random.seed(7)
        P = np.random.randn(20, 3)
        R_true = make_rotation('xyz', [55, 30, 10])
        Q = apply_xform(P, R_true, np.array([0.1, 0.2, 0.3]))
        xform = CameraRansacXform(threshold=0.05, iterations=500)
        R, _ = xform.compute_camera_xform(P, Q)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
