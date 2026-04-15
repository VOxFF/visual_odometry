"""
Tests for stereo/stereo_params_YAML.py

Covers: K matrix construction, baseline computation (Bug 2), focal length
averaging, R/T extraction, resolution, get_z_max, get_camera_params,
and error handling for bad inputs.
"""
import numpy as np
import pytest
from modules.stereo.stereo_params_YAML import StereoParamsYAML
from modules.stereo.stereo_interfaces import StereoParamsInterface


# ---------------------------------------------------------------------------
# Synthetic calibration YAML (Kalibr format)
# ---------------------------------------------------------------------------

# Baseline is mostly in X but has small Y and Z components — this is realistic
# and is exactly the scenario that exposes Bug 2.
_TX, _TY, _TZ = 0.1, 0.002, 0.001   # true baseline component vs norm

YAML_STANDARD = f"""
cam0:
  camera_model: pinhole
  intrinsics: [480.0, 480.0, 320.0, 240.0]
  distortion_model: radtan
  distortion_coeffs: [0.01, -0.02, 0.001, 0.0005]
  resolution: [640, 480]

cam1:
  camera_model: pinhole
  intrinsics: [482.0, 482.0, 318.0, 241.0]
  distortion_model: radtan
  distortion_coeffs: [0.012, -0.021, 0.0009, 0.0004]
  resolution: [640, 480]
  T_cn_cnm1:
    - [1.0, 0.0, 0.0, {_TX}]
    - [0.0, 1.0, 0.0, {_TY}]
    - [0.0, 0.0, 1.0, {_TZ}]
    - [0.0, 0.0, 0.0,  1.0]
"""

# YAML where both cameras have the same focal length (simplifies averaging check)
YAML_EQUAL_FX = """
cam0:
  intrinsics: [500.0, 500.0, 320.0, 240.0]
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  resolution: [640, 480]
cam1:
  intrinsics: [500.0, 500.0, 320.0, 240.0]
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  resolution: [640, 480]
  T_cn_cnm1:
    - [1.0, 0.0, 0.0, 0.1]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
"""


@pytest.fixture
def params():
    return StereoParamsYAML(YAML_STANDARD)


@pytest.fixture
def params_equal_fx():
    return StereoParamsYAML(YAML_EQUAL_FX)


# ---------------------------------------------------------------------------
# Intrinsic matrix construction
# ---------------------------------------------------------------------------

class TestIntrinsicMatrix:

    def test_K_l_shape(self, params):
        assert params.K_l.shape == (3, 3)

    def test_K_l_values(self, params):
        assert params.K_l[0, 0] == pytest.approx(480.0)   # fx
        assert params.K_l[1, 1] == pytest.approx(480.0)   # fy
        assert params.K_l[0, 2] == pytest.approx(320.0)   # cx
        assert params.K_l[1, 2] == pytest.approx(240.0)   # cy
        assert params.K_l[2, 2] == pytest.approx(1.0)
        assert params.K_l[0, 1] == pytest.approx(0.0)     # no skew

    def test_K_r_values(self, params):
        assert params.K_r[0, 0] == pytest.approx(482.0)
        assert params.K_r[0, 2] == pytest.approx(318.0)

    def test_K_l_is_upper_triangular(self, params):
        assert params.K_l[1, 0] == 0.0
        assert params.K_l[2, 0] == 0.0
        assert params.K_l[2, 1] == 0.0


# ---------------------------------------------------------------------------
# Extrinsics: R and T
# ---------------------------------------------------------------------------

class TestExtrinsics:

    def test_R_shape(self, params):
        assert params.R.shape == (3, 3)

    def test_R_is_identity_for_aligned_cameras(self, params):
        assert np.allclose(params.R, np.eye(3))

    def test_T_shape(self, params):
        assert params.T.shape == (3,)

    def test_T_values(self, params):
        assert params.T[0] == pytest.approx(_TX)
        assert params.T[1] == pytest.approx(_TY)
        assert params.T[2] == pytest.approx(_TZ)


# ---------------------------------------------------------------------------
# Baseline (Bug 2 is here)
# ---------------------------------------------------------------------------

class TestBaseline:

    def test_baseline_positive(self, params):
        assert params.get_baseline() > 0

    def test_baseline_equals_tx_not_norm(self, params):
        """
        Baseline must be the horizontal separation |T[0]|, NOT norm(T).
        This test FAILS with the current implementation (np.linalg.norm(T))
        and will PASS once Bug 2 is fixed.
        """
        correct_baseline = abs(_TX)
        wrong_baseline = np.linalg.norm([_TX, _TY, _TZ])
        assert wrong_baseline != pytest.approx(correct_baseline), \
            "Test setup error: T components chosen so norm == T_x; pick different values"
        assert params.get_baseline() == pytest.approx(correct_baseline, rel=1e-6)

    def test_pure_horizontal_baseline_unaffected(self, params_equal_fx):
        """When T is exactly horizontal (TY=TZ=0), norm(T)==|T_x| so both formulas agree."""
        assert params_equal_fx.get_baseline() == pytest.approx(0.1, rel=1e-6)


# ---------------------------------------------------------------------------
# Focal length averaging
# ---------------------------------------------------------------------------

class TestFocalLength:

    def test_focal_length_is_average_of_fx(self, params):
        expected = (480.0 + 482.0) / 2
        assert params.focal_length_px == pytest.approx(expected)

    def test_focal_length_equal_cameras(self, params_equal_fx):
        assert params_equal_fx.focal_length_px == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestResolution:

    def test_resolution_tuple(self, params):
        assert params.resolution == (640, 480)


# ---------------------------------------------------------------------------
# get_z_max
# ---------------------------------------------------------------------------

class TestZMax:

    def test_z_max_formula(self, params_equal_fx):
        """Z_max = f * baseline / d_min."""
        f = 500.0
        b = 0.1
        d_min = 1.0
        assert params_equal_fx.get_z_max(d_min) == pytest.approx(f * b / d_min)

    def test_z_max_scales_with_d_min(self, params_equal_fx):
        z1 = params_equal_fx.get_z_max(d_min=1.0)
        z2 = params_equal_fx.get_z_max(d_min=2.0)
        assert z1 == pytest.approx(2 * z2)


# ---------------------------------------------------------------------------
# get_camera_params
# ---------------------------------------------------------------------------

class TestGetCameraParams:

    def test_left_returns_K_l(self, params):
        cp = params.get_camera_params(StereoParamsInterface.StereoCamera.LEFT)
        assert np.allclose(cp.K, params.K_l)

    def test_right_returns_K_r(self, params):
        cp = params.get_camera_params(StereoParamsInterface.StereoCamera.RIGHT)
        assert np.allclose(cp.K, params.K_r)

    def test_invalid_camera_raises(self, params):
        with pytest.raises((ValueError, KeyError)):
            params.get_camera_params("invalid")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_missing_cam0_raises(self):
        bad_yaml = """
cam1:
  intrinsics: [500.0, 500.0, 320.0, 240.0]
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  resolution: [640, 480]
  T_cn_cnm1:
    - [1, 0, 0, 0.1]
    - [0, 1, 0, 0]
    - [0, 0, 1, 0]
    - [0, 0, 0, 1]
"""
        with pytest.raises(KeyError):
            StereoParamsYAML(bad_yaml)

    def test_missing_cam1_raises(self):
        bad_yaml = """
cam0:
  intrinsics: [500.0, 500.0, 320.0, 240.0]
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  resolution: [640, 480]
"""
        with pytest.raises(KeyError):
            StereoParamsYAML(bad_yaml)

    def test_invalid_file_extension_raises(self, tmp_path):
        bad_file = tmp_path / "params.txt"
        bad_file.write_text("cam0: {}")
        with pytest.raises(ValueError, match="Invalid file format"):
            StereoParamsYAML(str(bad_file))
