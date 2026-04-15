"""
Tests for stereo/stereo_rectification.py

Covers: rectification matrix availability, mask shape and validity,
image rectification output, and Bug 1 — P1[:3,:3] vs K_l mismatch.

All tests use a synthetic calibration with cameras that have different
principal points and non-zero distortion, so P1 is guaranteed to differ
from K_l after rectification.
"""
import numpy as np
import pytest
import cv2
from modules.stereo.stereo_params_YAML import StereoParamsYAML
from modules.stereo.stereo_rectification import StereoRectification

# ---------------------------------------------------------------------------
# Synthetic calibration — two cameras with different principal points
# and small but non-zero distortion. This guarantees P1[:3,:3] != K_l.
# ---------------------------------------------------------------------------

YAML_REALISTIC = """
cam0:
  intrinsics: [480.0, 480.0, 330.0, 250.0]
  distortion_coeffs: [-0.3, 0.1, 0.001, -0.0005]
  resolution: [640, 480]
cam1:
  intrinsics: [478.0, 478.0, 305.0, 245.0]
  distortion_coeffs: [-0.29, 0.098, 0.0009, -0.0004]
  resolution: [640, 480]
  T_cn_cnm1:
    - [ 0.9999,  0.0030, -0.0012,  0.1]
    - [-0.0030,  0.9999,  0.0008,  0.002]
    - [ 0.0012, -0.0008,  1.0000,  0.001]
    - [ 0.0,    0.0,     0.0,     1.0]
"""


@pytest.fixture(scope="module")
def params():
    return StereoParamsYAML(YAML_REALISTIC)


@pytest.fixture(scope="module")
def rectification(params):
    return StereoRectification(params)


# ---------------------------------------------------------------------------
# Rectification matrices
# ---------------------------------------------------------------------------

class TestRectificationMatrices:

    def test_all_matrices_available_after_init(self, rectification):
        R1, R2, P1, P2, Q = rectification.get_rectification_matrices()
        assert R1 is not None
        assert R2 is not None
        assert P1 is not None
        assert P2 is not None
        assert Q  is not None

    def test_R1_is_3x3(self, rectification):
        R1, *_ = rectification.get_rectification_matrices()
        assert R1.shape == (3, 3)

    def test_P1_is_3x4(self, rectification):
        _, _, P1, *_ = rectification.get_rectification_matrices()
        assert P1.shape == (3, 4)

    def test_P2_is_3x4(self, rectification):
        _, _, _, P2, _ = rectification.get_rectification_matrices()
        assert P2.shape == (3, 4)

    def test_Q_is_4x4(self, rectification):
        *_, Q = rectification.get_rectification_matrices()
        assert Q.shape == (4, 4)

    def test_R1_is_rotation_matrix(self, rectification):
        R1, *_ = rectification.get_rectification_matrices()
        assert np.allclose(R1 @ R1.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R1), 1.0, atol=1e-6)

    def test_P1_last_column_is_zero(self, rectification):
        """Left camera P1 has no translation — last column must be zero."""
        _, _, P1, *_ = rectification.get_rectification_matrices()
        assert np.allclose(P1[:, 3], 0.0, atol=1e-6)

    def test_P2_has_nonzero_tx(self, rectification):
        """Right camera P2 encodes the baseline in its last column."""
        _, _, _, P2, _ = rectification.get_rectification_matrices()
        assert P2[0, 3] != 0.0


# ---------------------------------------------------------------------------
# Rectification masks
# ---------------------------------------------------------------------------

class TestRectificationMasks:

    def test_returns_four_masks(self, rectification):
        masks = rectification.get_rectification_masks()
        assert len(masks) == 4

    def test_masks_are_bool(self, rectification):
        for mask in rectification.get_rectification_masks():
            assert mask.dtype == bool

    def test_mask_shape_matches_resolution(self, params, rectification):
        w, h = params.resolution
        for mask in rectification.get_rectification_masks():
            assert mask.shape == (h, w)

    def test_stereo_mask_is_subset_of_left_mask(self, rectification):
        stereo_mask, left_mask, right_mask, _ = rectification.get_rectification_masks()
        # stereo_mask must be True only where left_mask is also True
        assert np.all(stereo_mask[left_mask == False] == False)

    def test_stereo_mask_is_subset_of_right_mask(self, rectification):
        stereo_mask, _, right_mask, _ = rectification.get_rectification_masks()
        assert np.all(stereo_mask[right_mask == False] == False)

    def test_stereo_mask_has_valid_pixels(self, rectification):
        """A realistic calibration must leave some valid overlap area."""
        stereo_mask, *_ = rectification.get_rectification_masks()
        valid_fraction = np.sum(stereo_mask) / stereo_mask.size
        assert valid_fraction > 0.5   # expect >50% valid for a normal stereo rig


# ---------------------------------------------------------------------------
# Image rectification
# ---------------------------------------------------------------------------

class TestRectifyImages:

    def test_output_shape_unchanged(self, rectification):
        h, w = 480, 640
        img_l = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        img_r = np.random.randint(0, 255, (h, w), dtype=np.uint8)
        rect_l, rect_r = rectification.rectify_images(img_l, img_r)
        assert rect_l.shape == (h, w)
        assert rect_r.shape == (h, w)

    def test_left_only_rectification(self, rectification):
        img_l = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        rect_l, rect_r = rectification.rectify_images(img_l, None)
        assert rect_l is not None
        assert rect_r is None

    def test_right_only_rectification(self, rectification):
        img_r = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        rect_l, rect_r = rectification.rectify_images(None, img_r)
        assert rect_l is None
        assert rect_r is not None

    def test_uniform_image_stays_valid_in_mask(self, rectification):
        """A fully white image rectified must remain non-zero inside the stereo mask."""
        img = np.full((480, 640), 255, dtype=np.uint8)
        rect_l, _ = rectification.rectify_images(img, img)
        stereo_mask, *_ = rectification.get_rectification_masks()
        # All pixels inside the valid mask must be non-zero after rectification
        assert np.all(rect_l[stereo_mask] > 0)


# ---------------------------------------------------------------------------
# Bug 1: P1[:3,:3] vs K_l
# ---------------------------------------------------------------------------

class TestBug1RectifiedIntrinsics:

    def test_P1_intrinsics_differ_from_K_l(self, params, rectification):
        """
        After rectification, the effective left-camera intrinsics are P1[:3,:3],
        NOT the original K_l. This test documents the discrepancy.

        Keypoints3DXform currently uses K_l (pre-rectification) to back-project
        pixels from the rectified image — that is Bug 1.
        The fix is to use P1[:3,:3] instead.
        """
        _, _, P1, *_ = rectification.get_rectification_matrices()
        K_rect = P1[:3, :3]
        K_orig = params.K_l

        # They must differ — if this assertion fails, the calibration is trivial
        assert not np.allclose(K_rect, K_orig, atol=0.5), \
            "Calibration too trivial: K_l and rectified K are identical — choose a more realistic calibration"

        # Report the delta so the magnitude is visible in test output
        delta_cx = abs(K_orig[0, 2] - K_rect[0, 2])
        delta_cy = abs(K_orig[1, 2] - K_rect[1, 2])
        delta_fx = abs(K_orig[0, 0] - K_rect[0, 0])
        print(f"\nK_l:    fx={K_orig[0,0]:.2f}  cx={K_orig[0,2]:.2f}  cy={K_orig[1,2]:.2f}")
        print(f"K_rect: fx={K_rect[0,0]:.2f}  cx={K_rect[0,2]:.2f}  cy={K_rect[1,2]:.2f}")
        print(f"Delta:  Δfx={delta_fx:.2f}  Δcx={delta_cx:.2f}  Δcy={delta_cy:.2f}")

    def test_3d_backprojection_error_from_wrong_K(self, params, rectification):
        """
        Quantifies the 3D error introduced by using K_l instead of P1[:3,:3].

        For a point at the image centre with depth Z=3m, computes the
        difference in reconstructed X between the two intrinsics.
        A non-zero error confirms Bug 1 affects 3D point positions.
        """
        _, _, P1, *_ = rectification.get_rectification_matrices()
        K_rect = P1[:3, :3]
        K_orig = params.K_l

        # Use the principal point of the rectified image as test pixel
        u = K_rect[0, 2]
        v = K_rect[1, 2]
        Z = 3.0

        # Back-project with correct (rectified) intrinsics
        X_true = (u - K_rect[0, 2]) / K_rect[0, 0] * Z   # == 0 by construction
        Y_true = (v - K_rect[1, 2]) / K_rect[1, 1] * Z   # == 0 by construction

        # Back-project with wrong (original) intrinsics — as the code currently does
        X_wrong = (u - K_orig[0, 2]) / K_orig[0, 0] * Z
        Y_wrong = (v - K_orig[1, 2]) / K_orig[1, 1] * Z

        error = np.sqrt((X_wrong - X_true)**2 + (Y_wrong - Y_true)**2)
        print(f"\nAt rectified principal point (u={u:.1f}, v={v:.1f}), Z={Z}m:")
        print(f"  True XY:  ({X_true:.4f}, {Y_true:.4f})")
        print(f"  Wrong XY: ({X_wrong:.4f}, {Y_wrong:.4f})")
        print(f"  XY error: {error:.4f} m")

        assert error > 0.0, "Expected non-zero error due to K mismatch (Bug 1)"
