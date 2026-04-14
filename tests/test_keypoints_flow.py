"""
Tests for keypoints/keypoints_3d_flow.py

Covers: compute_2d_flow (displacement, bounds, indexing convention)
and compute_3d_flow (depth validity, output shape, valid mask).
"""
import numpy as np
import pytest
from keypoints.keypoints_3d import Keypoints3DXform
from keypoints.keypoints_3d_flow import Keypoints3DFlow
from tests.conftest import MockCameraParams


@pytest.fixture
def flow_tracker(camera_params, rectification_mask):
    xform = Keypoints3DXform(camera_params)
    return Keypoints3DFlow(camera_params, xform, rectification_mask)


def make_flow(H, W, du, dv):
    """Helper: constant optical flow of shape (2, H, W)."""
    flow = np.zeros((2, H, W), dtype=np.float32)
    flow[0] = du   # horizontal (u) component
    flow[1] = dv   # vertical (v) component
    return flow


# ---------------------------------------------------------------------------
# compute_2d_flow
# ---------------------------------------------------------------------------

class TestCompute2DFlow:

    def test_zero_flow_unchanged(self, flow_tracker):
        flow = make_flow(480, 640, 0, 0)
        pts = np.array([[100.0, 150.0], [300.0, 200.0]])
        result, valid = flow_tracker.compute_2d_flow(pts, flow)
        assert np.allclose(result, pts)
        assert np.all(valid)

    def test_constant_displacement(self, flow_tracker):
        du, dv = 10.0, -5.0
        flow = make_flow(480, 640, du, dv)
        pts = np.array([[100.0, 150.0], [300.0, 200.0]])
        result, valid = flow_tracker.compute_2d_flow(pts, flow)
        expected = pts + np.array([du, dv])
        assert np.allclose(result, expected)
        assert np.all(valid)

    def test_uv_indexing_convention(self, flow_tracker):
        """
        Verify (u, v) → flow[v, u] indexing:
        set flow[row=50, col=100] to (du=7, dv=3) and check
        that a keypoint at (u=100, v=50) gets displaced by (7, 3).
        """
        flow = make_flow(480, 640, 0, 0)
        flow[0, 50, 100] = 7.0   # du at row=50, col=100
        flow[1, 50, 100] = 3.0   # dv at row=50, col=100
        pts = np.array([[100.0, 50.0]])  # u=100, v=50
        result, valid = flow_tracker.compute_2d_flow(pts, flow)
        assert np.allclose(result[0], [107.0, 53.0])
        assert valid[0]

    def test_out_of_bounds_marked_invalid(self, flow_tracker):
        """Flow that pushes a point outside the image must mark it invalid."""
        flow = make_flow(480, 640, 700.0, 0)  # will push u past width=640
        pts = np.array([[300.0, 200.0]])
        _, valid = flow_tracker.compute_2d_flow(pts, flow)
        assert not valid[0]

    def test_in_bounds_marked_valid(self, flow_tracker):
        flow = make_flow(480, 640, 1.0, 1.0)
        pts = np.array([[100.0, 100.0]])
        _, valid = flow_tracker.compute_2d_flow(pts, flow)
        assert valid[0]

    def test_output_clipped_to_image(self, flow_tracker):
        """Even when marked invalid, coordinates must be clipped, not NaN/OOB."""
        flow = make_flow(480, 640, 1000.0, 1000.0)
        pts = np.array([[300.0, 200.0]])
        result, _ = flow_tracker.compute_2d_flow(pts, flow)
        assert result[0, 0] <= 639
        assert result[0, 1] <= 479


# ---------------------------------------------------------------------------
# compute_3d_flow
# ---------------------------------------------------------------------------

class TestCompute3DFlow:

    def test_output_shape(self, flow_tracker):
        depth = np.full((480, 640), 3.0)
        flow = make_flow(480, 640, 0, 0)
        pts = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]])
        pts_3d, valid = flow_tracker.compute_3d_flow(pts, depth, depth, flow)
        assert pts_3d.shape == (3, 3)
        assert valid.shape == (3,)

    def test_zero_flow_stationary_scene(self, flow_tracker):
        """Zero flow + same depth map: output 3D positions equal input 3D positions."""
        Z = 4.0
        depth = np.full((480, 640), Z)
        flow = make_flow(480, 640, 0, 0)
        pts = np.array([[320.0, 240.0], [100.0, 100.0]])

        pts_3d_out, valid = flow_tracker.compute_3d_flow(pts, depth, depth, flow)
        # Manually compute expected 3D positions
        xform = flow_tracker.keypoints_xform
        pts_3d_expected = xform.to_3d(pts, depth)

        assert np.all(valid)
        assert np.allclose(pts_3d_out[valid], pts_3d_expected[valid], atol=1e-6)

    def test_zero_depth_frame1_invalid(self, flow_tracker):
        """Zero depth in frame 1 at a keypoint → that keypoint invalid."""
        depth1 = np.zeros((480, 640))
        depth2 = np.full((480, 640), 3.0)
        flow = make_flow(480, 640, 0, 0)
        pts = np.array([[100.0, 100.0]])
        _, valid = flow_tracker.compute_3d_flow(pts, depth1, depth2, flow)
        assert not valid[0]

    def test_zero_depth_frame2_invalid(self, flow_tracker):
        """Zero depth in frame 2 at tracked position → that keypoint invalid."""
        depth1 = np.full((480, 640), 3.0)
        depth2 = np.zeros((480, 640))
        flow = make_flow(480, 640, 0, 0)
        pts = np.array([[100.0, 100.0]])
        _, valid = flow_tracker.compute_3d_flow(pts, depth1, depth2, flow)
        assert not valid[0]

    def test_out_of_bounds_flow_invalid(self, flow_tracker):
        """Flow that pushes point out of image → invalid."""
        depth = np.full((480, 640), 3.0)
        flow = make_flow(480, 640, 5000.0, 0)
        pts = np.array([[300.0, 200.0]])
        _, valid = flow_tracker.compute_3d_flow(pts, depth, depth, flow)
        assert not valid[0]

    def test_valid_mask_mixed(self, flow_tracker):
        """Mix of valid and invalid keypoints."""
        depth1 = np.full((480, 640), 3.0)
        depth2 = np.full((480, 640), 3.0)
        depth2[200, 110] = 0.0   # second point lands here after +10u flow

        flow = make_flow(480, 640, 10.0, 0)  # shift every point right by 10
        pts = np.array([
            [100.0, 100.0],   # valid: depth1>0, depth2[100,110]>0
            [100.0, 200.0],   # invalid: depth2[200, 110]=0
        ])
        _, valid = flow_tracker.compute_3d_flow(pts, depth1, depth2, flow)
        assert valid[0]
        assert not valid[1]
