"""
Tests for tools/pointsview

NOTE: requires system Python (python3) — pxr is installed for Python 3.10,
      not for obj_env (Python 3.8).
Run with:  python3 -m pytest tests/test_pointsview.py -v

Covers:
  h5_to_usda:
    - Stage time range, fps, up-axis
    - fps_override takes precedence over HDF5 attribute
    - Returns total landmark count
    - Prim hierarchy and path format
    - Points round-trip, widths, width value
    - Color: blue at frame 0, red at max frame, constant interpolation
    - Visibility: invisible before birth, visible from birth, frame-0 visible at t=0
    - sys.exit(1) on empty frames group

  main:
    - sys.exit(1) when h5 file not found
    - subprocess called with usdview and a .usda temp file
    - Temp file is deleted after usdview exits
"""

import importlib.util
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import h5py
import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

TOOLS_DIR = Path(__file__).parent.parent / 'tools'


# ── Module fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def pv():
    """Load tools/pointsview (no .py extension) as a Python module."""
    loader = SourceFileLoader('pointsview', str(TOOLS_DIR / 'pointsview'))
    spec   = importlib.util.spec_from_loader('pointsview', loader)
    mod    = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


# ── HDF5 fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def simple_h5(tmp_path):
    """Three-frame HDF5: frames 0, 5, 10 with 2, 1, 2 landmarks."""
    path = str(tmp_path / 'landmarks.h5')
    with h5py.File(path, 'w') as f:
        f.attrs['fps']   = 30.0
        f.attrs['total'] = 5
        grp = f.create_group('frames')
        grp.create_dataset('000000', data=np.array([[0., 0., 1.], [1., 0., 1.]], dtype=np.float32))
        grp.create_dataset('000005', data=np.array([[2., 0., 1.5]], dtype=np.float32))
        grp.create_dataset('000010', data=np.array([[3., 0., 2.], [3.5, 0., 2.]], dtype=np.float32))
    return path


@pytest.fixture
def empty_frames_h5(tmp_path):
    """HDF5 with an empty frames group."""
    path = str(tmp_path / 'empty.h5')
    with h5py.File(path, 'w') as f:
        f.attrs['fps']   = 30.0
        f.attrs['total'] = 0
        f.create_group('frames')
    return path


# ── h5_to_usda: stage properties ─────────────────────────────────────────────

class TestStageProperties:

    def test_start_time_is_zero(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert Usd.Stage.Open(out).GetStartTimeCode() == 0.0

    def test_end_time_equals_max_frame(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert Usd.Stage.Open(out).GetEndTimeCode() == 10.0

    def test_fps_from_h5_attribute(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert Usd.Stage.Open(out).GetTimeCodesPerSecond() == pytest.approx(30.0)

    def test_fps_override_takes_precedence(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, 60.0, 0.02)
        assert Usd.Stage.Open(out).GetTimeCodesPerSecond() == pytest.approx(60.0)

    def test_up_axis_is_y(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert UsdGeom.GetStageUpAxis(Usd.Stage.Open(out)) == UsdGeom.Tokens.y

    def test_returns_total_landmark_count(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        total = pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert total == 5

    def test_empty_frames_exits(self, pv, empty_frames_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        with pytest.raises(SystemExit) as exc:
            pv.h5_to_usda(empty_frames_h5, out, None, 0.02)
        assert exc.value.code == 1


# ── h5_to_usda: prim structure ────────────────────────────────────────────────

class TestPrimStructure:

    def test_world_xform_exists(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert Usd.Stage.Open(out).GetPrimAtPath('/World').IsValid()

    def test_landmarks_xform_exists(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert Usd.Stage.Open(out).GetPrimAtPath('/World/Landmarks').IsValid()

    def test_one_prim_per_frame(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        stage  = Usd.Stage.Open(out)
        prims  = [p for p in stage.Traverse()
                  if 'F0' in str(p.GetPath())]
        assert len(prims) == 3

    def test_prim_path_format(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        stage = Usd.Stage.Open(out)
        for frame_idx in (0, 5, 10):
            assert stage.GetPrimAtPath(f'/World/Landmarks/F{frame_idx:06d}').IsValid()

    def test_prims_are_points_type(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        stage = Usd.Stage.Open(out)
        prim  = stage.GetPrimAtPath('/World/Landmarks/F000000')
        assert prim.GetTypeName() == 'Points'

    def test_unrecorded_frame_has_no_prim(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert not Usd.Stage.Open(out).GetPrimAtPath('/World/Landmarks/F000007').IsValid()


# ── h5_to_usda: points and widths ────────────────────────────────────────────

class TestPointsAndWidths:

    def test_positions_round_trip(self, pv, tmp_path):
        path = str(tmp_path / 'pos.h5')
        pts  = np.array([[1.5, -0.3, 4.2], [0.0, 2.0, 1.0]], dtype=np.float32)
        with h5py.File(path, 'w') as f:
            f.attrs['fps'] = 30.0
            grp = f.create_group('frames')
            grp.create_dataset('000000', data=pts)
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(path, out, None, 0.02)
        stage  = Usd.Stage.Open(out)
        stored = UsdGeom.Points(
            stage.GetPrimAtPath('/World/Landmarks/F000000')
        ).GetPointsAttr().Get()
        assert len(stored) == 2
        for i, (x, y, z) in enumerate(pts):
            assert stored[i] == pytest.approx(Gf.Vec3f(float(x), float(y), float(z)), abs=1e-5)

    def test_widths_count_matches_point_count(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        stage = Usd.Stage.Open(out)
        prim  = UsdGeom.Points(stage.GetPrimAtPath('/World/Landmarks/F000000'))
        assert len(prim.GetWidthsAttr().Get()) == 2   # frame 0 has 2 points

    def test_width_value_from_argument(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, point_width=0.07)
        stage = Usd.Stage.Open(out)
        prim  = UsdGeom.Points(stage.GetPrimAtPath('/World/Landmarks/F000000'))
        for w in prim.GetWidthsAttr().Get():
            assert w == pytest.approx(0.07)


# ── h5_to_usda: color ────────────────────────────────────────────────────────

class TestColor:

    def _color(self, stage, frame_idx):
        prim = stage.GetPrimAtPath(f'/World/Landmarks/F{frame_idx:06d}')
        return UsdGeom.Points(prim).GetDisplayColorAttr().Get()[0]

    def test_first_frame_is_blue(self, pv, simple_h5, tmp_path):
        out   = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        color = self._color(Usd.Stage.Open(out), 0)
        assert color[0] == pytest.approx(0.0, abs=1e-5)   # R = 0
        assert color[2] == pytest.approx(1.0, abs=1e-5)   # B = 1

    def test_last_frame_is_red(self, pv, simple_h5, tmp_path):
        out   = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        color = self._color(Usd.Stage.Open(out), 10)
        assert color[0] == pytest.approx(1.0, abs=1e-5)   # R = 1
        assert color[2] == pytest.approx(0.0, abs=1e-5)   # B = 0

    def test_red_increases_blue_decreases_with_frame(self, pv, simple_h5, tmp_path):
        out   = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        stage = Usd.Stage.Open(out)
        r0,  b0  = self._color(stage,  0)[0], self._color(stage,  0)[2]
        r5,  b5  = self._color(stage,  5)[0], self._color(stage,  5)[2]
        r10, b10 = self._color(stage, 10)[0], self._color(stage, 10)[2]
        assert r0 < r5 < r10
        assert b0 > b5 > b10

    def test_color_interpolation_is_constant(self, pv, simple_h5, tmp_path):
        out = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        stage  = Usd.Stage.Open(out)
        prim   = stage.GetPrimAtPath('/World/Landmarks/F000000')
        cpv    = UsdGeom.Points(prim).GetDisplayColorPrimvar()
        assert cpv.GetInterpolation() == UsdGeom.Tokens.constant
        assert len(cpv.Get()) == 1


# ── h5_to_usda: visibility ────────────────────────────────────────────────────

class TestVisibility:

    def _vis(self, stage, frame_idx, time):
        prim = stage.GetPrimAtPath(f'/World/Landmarks/F{frame_idx:06d}')
        return UsdGeom.Imageable(prim).GetVisibilityAttr().Get(time)

    def test_nonzero_frame_invisible_before_birth(self, pv, simple_h5, tmp_path):
        out   = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert self._vis(Usd.Stage.Open(out), 5, 0) == UsdGeom.Tokens.invisible

    def test_nonzero_frame_visible_at_birth(self, pv, simple_h5, tmp_path):
        out   = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert self._vis(Usd.Stage.Open(out), 5, 5) == UsdGeom.Tokens.inherited

    def test_nonzero_frame_stays_visible_after_birth(self, pv, simple_h5, tmp_path):
        out   = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert self._vis(Usd.Stage.Open(out), 5, 999) == UsdGeom.Tokens.inherited

    def test_frame_zero_visible_at_t0(self, pv, simple_h5, tmp_path):
        out   = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        assert self._vis(Usd.Stage.Open(out), 0, 0) == UsdGeom.Tokens.inherited

    def test_nonzero_frame_has_two_visibility_samples(self, pv, simple_h5, tmp_path):
        out   = str(tmp_path / 'out.usda')
        pv.h5_to_usda(simple_h5, out, None, 0.02)
        stage  = Usd.Stage.Open(out)
        prim   = stage.GetPrimAtPath('/World/Landmarks/F000005')
        samples = UsdGeom.Imageable(prim).GetVisibilityAttr().GetTimeSamples()
        assert set(samples) == {0.0, 5.0}


# ── main ──────────────────────────────────────────────────────────────────────

class TestMain:

    def test_missing_file_exits(self, pv, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'argv', ['pointsview', str(tmp_path / 'nope.h5')])
        with pytest.raises(SystemExit) as exc:
            pv.main()
        assert exc.value.code == 1

    def test_calls_usdview_with_usda(self, pv, simple_h5, monkeypatch):
        calls = []
        monkeypatch.setattr(pv.subprocess, 'run',
                            lambda cmd, **kw: calls.append(cmd))
        monkeypatch.setattr(sys, 'argv', ['pointsview', simple_h5])
        pv.main()
        assert len(calls) == 1
        assert calls[0][-1].endswith('.usda')

    def test_temp_file_deleted_after_exit(self, pv, simple_h5, monkeypatch):
        captured = []
        def mock_run(cmd, **kw):
            captured.append(cmd[-1])   # path to temp .usda
        monkeypatch.setattr(pv.subprocess, 'run', mock_run)
        monkeypatch.setattr(sys, 'argv', ['pointsview', simple_h5])
        pv.main()
        assert not os.path.exists(captured[0])

    def test_temp_file_deleted_even_if_usdview_fails(self, pv, simple_h5, monkeypatch):
        captured = []
        def mock_run_fail(cmd, **kw):
            captured.append(cmd[-1])
            raise subprocess.CalledProcessError(1, cmd)
        monkeypatch.setattr(pv.subprocess, 'run', mock_run_fail)
        monkeypatch.setattr(sys, 'argv', ['pointsview', simple_h5])
        with pytest.raises(subprocess.CalledProcessError):
            pv.main()
        assert not os.path.exists(captured[0])
