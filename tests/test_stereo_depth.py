"""
Tests for stereo/stereo_depth.py

Covers: depth formula correctness, disparity sign handling,
invalid-disparity masking, edge cases.
"""
import numpy as np
import pytest
from modules.stereo.stereo_depth import StereoDepth


@pytest.fixture
def solver(stereo_params):
    return StereoDepth(stereo_params)


class TestStereoDepth:

    def test_known_value(self, solver):
        """depth = (f * baseline) / disparity — verify with known numbers."""
        # f=500, b=0.1 → depth = 50 / disp
        disp = np.array([[10.0]])
        depth = solver.compute_depth(disp)
        assert depth[0, 0] == pytest.approx(50.0 / 10.0, rel=1e-6)

    def test_zero_disparity_gives_zero_depth(self, solver):
        """Zero disparity is invalid — must return 0."""
        disp = np.zeros((4, 4))
        depth = solver.compute_depth(disp)
        assert np.all(depth == 0.0)

    def test_negative_disparity_handled(self, solver):
        """Negative disparity (RAFT-Stereo convention) must produce positive depth."""
        disp = np.array([[-10.0]])
        depth = solver.compute_depth(disp)
        assert depth[0, 0] == pytest.approx(50.0 / 10.0, rel=1e-6)

    def test_mixed_valid_invalid(self, solver):
        """Zero entries give 0 depth; non-zero entries give valid depth."""
        disp = np.array([[5.0, 0.0], [0.0, 25.0]])
        depth = solver.compute_depth(disp)
        assert depth[0, 0] == pytest.approx(50.0 / 5.0)
        assert depth[0, 1] == 0.0
        assert depth[1, 0] == 0.0
        assert depth[1, 1] == pytest.approx(50.0 / 25.0)

    def test_output_shape_preserved(self, solver):
        disp = np.random.uniform(1, 10, (480, 640))
        depth = solver.compute_depth(disp)
        assert depth.shape == (480, 640)

    def test_none_disparity_raises(self, solver):
        with pytest.raises(ValueError):
            solver.compute_depth(None)

    def test_depth_inversely_proportional_to_disparity(self, solver):
        """Doubling disparity must halve depth."""
        disp1 = np.array([[10.0]])
        disp2 = np.array([[20.0]])
        d1 = solver.compute_depth(disp1)[0, 0]
        d2 = solver.compute_depth(disp2)[0, 0]
        assert d1 == pytest.approx(2 * d2, rel=1e-6)

    def test_invalid_params_raises(self, stereo_params):
        """Zero focal length or baseline must be rejected at construction."""
        from tests.conftest import MockStereoParams
        with pytest.raises(ValueError):
            StereoDepth(MockStereoParams(focal_length_px=0.0, baseline=0.1))
        with pytest.raises(ValueError):
            StereoDepth(MockStereoParams(focal_length_px=500.0, baseline=0.0))
