"""
Tests for modules/landmarks/h5_export.py

Covers:
  - record_new: empty input ignored, dtype coercion, multi-frame accumulation
  - write: file created, root attributes (fps, total), no-data early exit
  - HDF5 structure: frames group exists, one dataset per frame, correct key format
  - Dataset contents: positions round-trip, shape, dtype
  - Compression: datasets are compressed (gzip)
  - Edge cases: single frame, single point, large frame index, duplicate frame calls
"""

import numpy as np
import pytest
import h5py

from modules.landmarks.h5_export import LandmarkH5Exporter


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_exporter(*frame_data):
    """Build a pre-loaded exporter from (frame_idx, [[x,y,z], ...]) pairs."""
    exp = LandmarkH5Exporter()
    for frame_idx, pts in frame_data:
        exp.record_new(frame_idx, [np.array(p, dtype=np.float32) for p in pts])
    return exp


def open_h5(path):
    return h5py.File(str(path), 'r')


# ── record_new ────────────────────────────────────────────────────────────────

class TestRecordNew:

    def test_empty_list_not_stored(self):
        exp = LandmarkH5Exporter()
        exp.record_new(0, [])
        assert len(exp._frames) == 0

    def test_single_frame_stored(self):
        exp = LandmarkH5Exporter()
        exp.record_new(5, [np.array([1., 2., 3.])])
        assert len(exp._frames) == 1
        frame_idx, xyzs = exp._frames[0]
        assert frame_idx == 5
        assert xyzs.shape == (1, 3)

    def test_multiple_frames_accumulate(self):
        exp = LandmarkH5Exporter()
        exp.record_new(0,  [np.array([0., 0., 0.])])
        exp.record_new(10, [np.array([1., 1., 1.]), np.array([2., 2., 2.])])
        assert len(exp._frames) == 2

    def test_stored_as_float32(self):
        exp = LandmarkH5Exporter()
        exp.record_new(0, [np.array([1., 2., 3.], dtype=np.float64)])
        _, xyzs = exp._frames[0]
        assert xyzs.dtype == np.float32

    def test_multiple_points_shape(self):
        pts = [np.array([float(i), 0., 1.]) for i in range(10)]
        exp = LandmarkH5Exporter()
        exp.record_new(3, pts)
        _, xyzs = exp._frames[0]
        assert xyzs.shape == (10, 3)


# ── write: file and root attributes ──────────────────────────────────────────

class TestWriteAttributes:

    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "landmarks.h5")
        make_exporter((0, [[1., 2., 3.]])).write(out)
        assert (tmp_path / "landmarks.h5").exists()

    def test_no_data_does_not_create_file(self, tmp_path):
        out = tmp_path / "landmarks.h5"
        LandmarkH5Exporter().write(str(out))
        assert not out.exists()

    def test_fps_attribute_default(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter((0, [[0., 0., 0.]])).write(out)
        with open_h5(out) as f:
            assert f.attrs['fps'] == pytest.approx(30.0)

    def test_fps_attribute_custom(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter((0, [[0., 0., 0.]])).write(out, fps=60.0)
        with open_h5(out) as f:
            assert f.attrs['fps'] == pytest.approx(60.0)

    def test_total_attribute_counts_all_landmarks(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter(
            (0,  [[0., 0., 0.], [1., 0., 0.]]),   # 2
            (5,  [[2., 0., 0.]]),                  # 1
            (10, [[3., 0., 0.], [4., 0., 0.], [5., 0., 0.]]),  # 3
        ).write(out)
        with open_h5(out) as f:
            assert f.attrs['total'] == 6


# ── HDF5 structure ────────────────────────────────────────────────────────────

class TestHdf5Structure:

    def test_frames_group_exists(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter((0, [[0., 0., 0.]])).write(out)
        with open_h5(out) as f:
            assert 'frames' in f

    def test_one_dataset_per_frame(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter(
            (0,  [[0., 0., 0.]]),
            (5,  [[1., 0., 0.]]),
            (10, [[2., 0., 0.]]),
        ).write(out)
        with open_h5(out) as f:
            assert len(f['frames']) == 3

    def test_dataset_key_format(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter((42, [[0., 0., 0.]])).write(out)
        with open_h5(out) as f:
            assert '000042' in f['frames']

    def test_only_recorded_frames_present(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter((0, [[0., 0., 0.]]), (100, [[1., 0., 0.]])).write(out)
        with open_h5(out) as f:
            assert '000050' not in f['frames']


# ── Dataset contents ──────────────────────────────────────────────────────────

class TestDatasetContents:

    def test_positions_round_trip(self, tmp_path):
        out = str(tmp_path / "out.h5")
        pts = [[1.5, -0.3, 4.2], [0.0, 2.0, 1.0]]
        make_exporter((0, pts)).write(out)
        with open_h5(out) as f:
            stored = f['frames']['000000'][:]
        assert stored.shape == (2, 3)
        assert stored == pytest.approx(np.array(pts, dtype=np.float32), abs=1e-6)

    def test_dataset_dtype_is_float32(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter((0, [[1., 2., 3.]])).write(out)
        with open_h5(out) as f:
            assert f['frames']['000000'].dtype == np.float32

    def test_dataset_shape_n_by_3(self, tmp_path):
        out = str(tmp_path / "out.h5")
        pts = [[float(i), 0., 1.] for i in range(7)]
        make_exporter((0, pts)).write(out)
        with open_h5(out) as f:
            assert f['frames']['000000'].shape == (7, 3)

    def test_single_point(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter((0, [[3.14, 2.72, 1.41]])).write(out)
        with open_h5(out) as f:
            stored = f['frames']['000000'][:]
        assert stored.shape == (1, 3)
        assert stored[0] == pytest.approx([3.14, 2.72, 1.41], abs=1e-4)

    def test_multiple_frames_independent(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter(
            (0, [[0., 0., 0.]]),
            (5, [[9., 9., 9.]]),
        ).write(out)
        with open_h5(out) as f:
            assert f['frames']['000000'][0, 0] == pytest.approx(0.0)
            assert f['frames']['000005'][0, 0] == pytest.approx(9.0)


# ── Compression ───────────────────────────────────────────────────────────────

class TestCompression:

    def test_datasets_are_gzip_compressed(self, tmp_path):
        out = str(tmp_path / "out.h5")
        pts = [[float(i), 0., 1.] for i in range(50)]
        make_exporter((0, pts)).write(out)
        with open_h5(out) as f:
            ds = f['frames']['000000']
            assert ds.compression == 'gzip'

    def test_compressed_file_smaller_than_raw(self, tmp_path):
        """Repeated coordinates compress well — file should be smaller than raw floats."""
        out = str(tmp_path / "out.h5")
        pts = [[1.0, 2.0, 3.0]] * 1000
        make_exporter((0, pts)).write(out)
        raw_bytes = 1000 * 3 * 4   # 1000 points × 3 floats × 4 bytes
        assert (tmp_path / "out.h5").stat().st_size < raw_bytes


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_large_frame_index(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter((999999, [[0., 0., 0.]])).write(out)
        with open_h5(out) as f:
            assert '999999' in f['frames']

    def test_duplicate_record_calls_same_frame(self, tmp_path):
        """Two record_new calls at the same frame index both appear in the file."""
        exp = LandmarkH5Exporter()
        exp.record_new(3, [np.array([0., 0., 0.])])
        exp.record_new(3, [np.array([1., 1., 1.])])
        assert len(exp._frames) == 2

    def test_single_frame_only(self, tmp_path):
        out = str(tmp_path / "out.h5")
        make_exporter((0, [[1., 2., 3.]])).write(out)
        with open_h5(out) as f:
            assert len(f['frames']) == 1
            assert f.attrs['total'] == 1
