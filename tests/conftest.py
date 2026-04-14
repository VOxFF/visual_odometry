"""
Shared fixtures for the visual odometry test suite.
"""
import numpy as np
import pytest
from stereo.stereo_interfaces import CameraParametersInterface, StereoParamsInterface


class MockCameraParams(CameraParametersInterface):
    """Minimal camera parameters for testing — no real calibration file needed."""

    def __init__(self, K: np.ndarray, resolution=(640, 480)):
        self.K = K
        self.D = np.zeros(5)
        self.resolution = resolution

    def get_intrinsics(self):
        return self.K

    def get_distortion_coeffs(self):
        return self.D

    def get_resolution(self):
        return self.resolution


class MockStereoParams(StereoParamsInterface):
    """Minimal stereo params for testing StereoDepth."""

    def __init__(self, focal_length_px: float, baseline: float):
        self.focal_length_px = focal_length_px
        self._baseline = baseline

    def load_params(self, input_data):
        pass

    def get_intrinsics(self):
        return {}

    def get_baseline(self):
        return self._baseline

    def get_z_max(self, d_min=1.0):
        return self.focal_length_px * self._baseline / d_min


@pytest.fixture
def K_standard():
    """A typical 640x480 camera intrinsic matrix."""
    return np.array([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ])


@pytest.fixture
def camera_params(K_standard):
    return MockCameraParams(K_standard)


@pytest.fixture
def stereo_params():
    return MockStereoParams(focal_length_px=500.0, baseline=0.1)


@pytest.fixture
def rectification_mask():
    """All-valid 480x640 mask."""
    return np.ones((480, 640), dtype=bool)
