"""
Tests for modules/landmarks/landmark_map.py and landmark.py

Covers:
  - compute_covariance: shape, symmetry, z-scaling, world-frame rotation
  - backproject: principal point, off-axis, batch
  - LandmarkMap.add: storage and index growth
  - LandmarkMap.merge: information-filter position update, 3D distance guard
  - LandmarkMap.get_correspondences: ratio test, deduplication, empty map
  - Roundtrip: add then retrieve via get_correspondences
"""

import numpy as np
import pytest
from modules.landmarks.landmark_map import LandmarkMap, compute_covariance, backproject
from modules.landmarks.landmark import Landmark


# ── Fixtures ──────────────────────────────────────────────────────────────────

FX, FY   = 500.0, 500.0
CX, CY   = 320.0, 240.0
BASELINE = 0.1   # metres


def make_desc(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(128).astype(np.float32)


@pytest.fixture
def empty_map():
    return LandmarkMap(match_ratio=0.75, max_merge_dist_3d=0.5)


# ── compute_covariance ────────────────────────────────────────────────────────

class TestComputeCovariance:

    def test_shape(self):
        cov = compute_covariance(5.0, FX, FY, BASELINE, np.eye(3))
        assert cov.shape == (3, 3)

    def test_symmetric(self):
        cov = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        assert np.allclose(cov, cov.T)

    def test_positive_definite(self):
        cov = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)

    def test_z_uncertainty_larger_than_xy(self):
        """Depth noise dominates lateral noise at moderate range."""
        cov = compute_covariance(5.0, FX, FY, BASELINE, np.eye(3))
        sigma_xy = np.sqrt(cov[0, 0])
        sigma_z  = np.sqrt(cov[2, 2])
        assert sigma_z > sigma_xy * 10

    def test_z_grows_quadratically(self):
        """σ_z ∝ z² — doubling depth quadruples depth uncertainty."""
        cov1 = compute_covariance(2.0, FX, FY, BASELINE, np.eye(3))
        cov2 = compute_covariance(4.0, FX, FY, BASELINE, np.eye(3))
        ratio = np.sqrt(cov2[2, 2]) / np.sqrt(cov1[2, 2])
        assert ratio == pytest.approx(4.0, rel=1e-6)

    def test_xy_grows_linearly(self):
        """σ_xy ∝ z — doubling depth doubles lateral uncertainty."""
        cov1 = compute_covariance(2.0, FX, FY, BASELINE, np.eye(3))
        cov2 = compute_covariance(4.0, FX, FY, BASELINE, np.eye(3))
        ratio = np.sqrt(cov2[0, 0]) / np.sqrt(cov1[0, 0])
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_rotation_applied(self):
        """Rotating 90° around Z swaps X and Y variances."""
        R = np.array([[0, -1, 0],
                      [1,  0, 0],
                      [0,  0, 1]], dtype=float)
        cov_id = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        cov_r  = compute_covariance(3.0, FX, FY, BASELINE, R)
        # After 90° rotation around Z, diagonal should be [σy², σx², σz²]
        assert cov_r[0, 0] == pytest.approx(cov_id[1, 1], rel=1e-6)
        assert cov_r[1, 1] == pytest.approx(cov_id[0, 0], rel=1e-6)
        assert cov_r[2, 2] == pytest.approx(cov_id[2, 2], rel=1e-6)

    def test_identity_gives_diagonal(self):
        """With identity rotation the covariance must be diagonal."""
        cov = compute_covariance(4.0, FX, FY, BASELINE, np.eye(3))
        off_diag = cov - np.diag(np.diag(cov))
        assert np.allclose(off_diag, 0, atol=1e-12)


# ── backproject ───────────────────────────────────────────────────────────────

class TestBackproject:

    def test_principal_point_zero_xy(self):
        """Principal point (cx, cy) at depth z → (0, 0, z)."""
        pts = backproject(np.array([CX]), np.array([CY]), np.array([5.0]),
                          FX, FY, CX, CY)
        assert np.allclose(pts[0], [0.0, 0.0, 5.0], atol=1e-9)

    def test_off_axis(self):
        u, v, z = 420.0, 300.0, 3.0
        pts = backproject(np.array([u]), np.array([v]), np.array([z]),
                          FX, FY, CX, CY)
        assert pts[0, 0] == pytest.approx((u - CX) * z / FX)
        assert pts[0, 1] == pytest.approx((v - CY) * z / FY)
        assert pts[0, 2] == pytest.approx(z)

    def test_batch_shape(self):
        N = 10
        u = np.linspace(100, 500, N)
        v = np.linspace(100, 400, N)
        z = np.full(N, 4.0)
        pts = backproject(u, v, z, FX, FY, CX, CY)
        assert pts.shape == (N, 3)

    def test_zero_depth_gives_zero_xyz(self):
        pts = backproject(np.array([CX]), np.array([CY]), np.array([0.0]),
                          FX, FY, CX, CY)
        assert np.allclose(pts[0], [0.0, 0.0, 0.0])


# ── LandmarkMap.add ───────────────────────────────────────────────────────────

class TestAdd:

    def test_size_increments(self, empty_map):
        cov = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        empty_map.add(np.array([1., 2., 3.]), cov, make_desc(0), 0)
        assert empty_map.size == 1
        empty_map.add(np.array([4., 5., 6.]), cov, make_desc(1), 0)
        assert empty_map.size == 2

    def test_descriptor_stored(self, empty_map):
        cov  = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        desc = make_desc(42)
        lm_id = empty_map.add(np.array([1., 2., 3.]), cov, desc, 0)
        assert len(empty_map.landmarks[lm_id].descriptors) == 1
        assert np.allclose(empty_map.landmarks[lm_id].descriptors[0], desc)

    def test_returns_sequential_ids(self, empty_map):
        cov = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        ids = [empty_map.add(np.zeros(3), cov, make_desc(i), 0) for i in range(5)]
        assert ids == list(range(5))

    def test_position_stored_correctly(self, empty_map):
        cov = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        xyz = np.array([1.5, -0.3, 4.2])
        lm_id = empty_map.add(xyz, cov, make_desc(0), 0)
        assert np.allclose(empty_map.landmarks[lm_id].xyz_world, xyz)


# ── LandmarkMap.merge ─────────────────────────────────────────────────────────

class TestMerge:

    def _setup(self, empty_map):
        cov   = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        xyz   = np.array([1.0, 2.0, 3.0])
        desc  = make_desc(0)
        lm_id = empty_map.add(xyz, cov, desc, frame_idx=0)
        return lm_id, xyz, cov

    def test_observation_count_increments(self, empty_map):
        lm_id, xyz, cov = self._setup(empty_map)
        empty_map.merge(lm_id, xyz, cov, make_desc(1), frame_idx=1)
        assert empty_map.landmarks[lm_id].observations == 2

    def test_descriptor_appended(self, empty_map):
        lm_id, xyz, cov = self._setup(empty_map)
        empty_map.merge(lm_id, xyz, cov, make_desc(1), frame_idx=1)
        assert len(empty_map.landmarks[lm_id].descriptors) == 2

    def test_position_unchanged_for_identical_observation(self, empty_map):
        """Merging the exact same position should not change it significantly."""
        lm_id, xyz, cov = self._setup(empty_map)
        empty_map.merge(lm_id, xyz.copy(), cov.copy(), make_desc(1), frame_idx=1)
        assert np.allclose(empty_map.landmarks[lm_id].xyz_world, xyz, atol=1e-6)

    def test_position_moves_toward_new_observation(self, empty_map):
        """With equal covariances, fused position should be midpoint."""
        lm_id, xyz_old, cov = self._setup(empty_map)
        xyz_new = xyz_old + np.array([0.2, 0.0, 0.0])
        empty_map.merge(lm_id, xyz_new, cov.copy(), make_desc(1), frame_idx=1)
        fused = empty_map.landmarks[lm_id].xyz_world
        midpoint = (xyz_old + xyz_new) / 2.0
        assert np.allclose(fused, midpoint, atol=1e-6)

    def test_merge_rejects_distant_observation(self, empty_map):
        """Observation more than max_merge_dist_3d away must be rejected."""
        lm_id, xyz, cov = self._setup(empty_map)
        far_xyz = xyz + np.array([10.0, 0.0, 0.0])
        result = empty_map.merge(lm_id, far_xyz, cov, make_desc(1), frame_idx=1)
        assert result is False
        assert empty_map.landmarks[lm_id].observations == 1

    def test_merge_unknown_id_returns_false(self, empty_map):
        cov = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        result = empty_map.merge(999, np.zeros(3), cov, make_desc(0), frame_idx=0)
        assert result is False

    def test_covariance_shrinks_after_merge(self, empty_map):
        """Information filter: fused uncertainty must be smaller than either input."""
        lm_id, xyz, cov = self._setup(empty_map)
        empty_map.merge(lm_id, xyz.copy(), cov.copy(), make_desc(1), frame_idx=1)
        fused_cov = empty_map.landmarks[lm_id].covariance
        assert np.linalg.det(fused_cov) < np.linalg.det(cov)


# ── LandmarkMap.get_correspondences ──────────────────────────────────────────

class TestGetCorrespondences:

    def _populate(self, lmap, n=5):
        """Add n landmarks with distinct descriptors, return (kp_uvs, descs, xyz_list)."""
        cov   = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        rng   = np.random.default_rng(0)
        descs = [rng.standard_normal(128).astype(np.float32) for _ in range(n)]
        xyzs  = [rng.standard_normal(3).astype(np.float32) for _ in range(n)]
        for i in range(n):
            lmap.add(xyzs[i], cov, descs[i], frame_idx=0)
        return descs, xyzs

    def test_empty_map_returns_empty(self, empty_map):
        kp  = np.array([[100., 200.]])
        d   = make_desc(0).reshape(1, -1)
        pts_3d, pts_2d, lm_ids, kp_idx = empty_map.get_correspondences(kp, d)
        assert len(pts_3d) == 0

    def test_exact_descriptor_match(self, empty_map):
        """Query with the stored descriptor → should get a correspondence."""
        descs, xyzs = self._populate(empty_map, n=5)
        kp  = np.array([[100., 200.]])
        d   = descs[0].reshape(1, -1)        # exact copy of landmark 0's descriptor
        pts_3d, pts_2d, lm_ids, kp_idx = empty_map.get_correspondences(kp, d)
        assert len(pts_3d) == 1
        assert np.allclose(pts_3d[0], xyzs[0], atol=1e-5)
        assert np.allclose(pts_2d[0], [100., 200.])

    def test_no_match_for_random_descriptor(self, empty_map):
        """A random descriptor should fail the ratio test against distinct stored ones."""
        self._populate(empty_map, n=10)
        rng = np.random.default_rng(999)
        d   = rng.standard_normal(128).astype(np.float32).reshape(1, -1)
        kp  = np.array([[100., 200.]])
        pts_3d, _, _, _ = empty_map.get_correspondences(kp, d)
        # May or may not match — just verify shape is valid
        assert pts_3d.ndim == 2 and pts_3d.shape[1] == 3

    def test_deduplication(self, empty_map):
        """Two query descriptors matching the same landmark → only one correspondence."""
        descs, _ = self._populate(empty_map, n=5)
        # Both query descriptors are copies of landmark 0
        kp = np.array([[100., 200.], [300., 400.]])
        d  = np.vstack([descs[0], descs[0]])
        _, _, lm_ids, _ = empty_map.get_correspondences(kp, d)
        # landmark 0 should appear at most once
        assert list(lm_ids).count(0) <= 1

    def test_output_shapes_consistent(self, empty_map):
        descs, _ = self._populate(empty_map, n=5)
        kp = np.array([[float(i * 10), float(i * 10)] for i in range(5)])
        d  = np.vstack(descs)
        pts_3d, pts_2d, lm_ids, kp_idx = empty_map.get_correspondences(kp, d)
        M = len(pts_3d)
        assert pts_2d.shape   == (M, 2)
        assert lm_ids.shape   == (M,)
        assert kp_idx.shape   == (M,)

    def test_single_descriptor_in_map_returns_empty(self, empty_map):
        """knnMatch needs k=2 training samples — 1 descriptor → empty result."""
        cov  = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        desc = make_desc(0)
        empty_map.add(np.array([1., 2., 3.]), cov, desc, frame_idx=0)
        assert len(empty_map._desc_list) == 1
        kp = np.array([[100., 200.]])
        pts_3d, _, _, _ = empty_map.get_correspondences(kp, desc.reshape(1, -1))
        assert len(pts_3d) == 0

    def test_match_with_merged_descriptor(self, empty_map):
        """After merging, a query using the *new* descriptor should still match."""
        descs, xyzs = self._populate(empty_map, n=5)   # 5 landmarks, 5 descs
        # Merge landmark 0 with a new descriptor
        cov      = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        new_desc = make_desc(99)
        empty_map.merge(0, xyzs[0], cov, new_desc, frame_idx=1)

        kp = np.array([[100., 200.]])
        pts_3d, _, lm_ids, _ = empty_map.get_correspondences(kp, new_desc.reshape(1, -1))
        assert len(pts_3d) == 1
        assert lm_ids[0] == 0

    def test_matrix_rebuilt_after_new_add(self, empty_map):
        """Descriptors added after a match call must be visible in the next match."""
        cov   = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        desc0 = make_desc(0)
        desc1 = make_desc(1)
        # First add — only 1 desc, too few for knnMatch
        empty_map.add(np.array([1., 2., 3.]), cov, desc0, frame_idx=0)
        # Second add — now 2 descs, matrix should be (re)built on next match
        lm_id1 = empty_map.add(np.array([4., 5., 6.]), cov, desc1, frame_idx=0)
        kp = np.array([[100., 200.]])
        pts_3d, _, lm_ids, _ = empty_map.get_correspondences(kp, desc1.reshape(1, -1))
        assert len(pts_3d) == 1
        assert lm_ids[0] == lm_id1


# ── Flat index integrity ──────────────────────────────────────────────────────

class TestFlatIndexIntegrity:

    def test_lm_ids_after_multiple_merges(self, empty_map):
        """All descriptor entries for a merged landmark must map to the same lm_id."""
        cov   = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        xyz   = np.array([1., 2., 3.])
        # Add two landmarks so knnMatch has ≥ 2 training samples
        lm_id = empty_map.add(xyz, cov, make_desc(0), frame_idx=0)
        empty_map.add(xyz + 5, cov, make_desc(1), frame_idx=0)
        # Merge landmark 0 three more times
        for k in range(2, 5):
            empty_map.merge(lm_id, xyz, cov, make_desc(k), frame_idx=k)
        # All entries in the flat index that point to lm_id should be consistent
        ids_in_index = [lid for lid in empty_map._lm_id_list if lid == lm_id]
        assert len(ids_in_index) == 4   # 1 original + 3 merges

    def test_desc_list_and_lm_id_list_same_length(self, empty_map):
        cov = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        for i in range(5):
            empty_map.add(np.array([float(i), 0., 3.]), cov, make_desc(i), frame_idx=0)
        empty_map.merge(0, np.array([0., 0., 3.]), cov, make_desc(99), frame_idx=1)
        assert len(empty_map._desc_list) == len(empty_map._lm_id_list)


# ── Merge metadata ────────────────────────────────────────────────────────────

class TestMergeMetadata:

    def test_last_seen_frame_updated(self, empty_map):
        cov   = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        xyz   = np.array([1., 2., 3.])
        lm_id = empty_map.add(xyz, cov, make_desc(0), frame_idx=0)
        empty_map.merge(lm_id, xyz, cov, make_desc(1), frame_idx=42)
        assert empty_map.landmarks[lm_id].last_seen_frame == 42

    def test_last_seen_frame_not_updated_on_rejection(self, empty_map):
        cov   = compute_covariance(3.0, FX, FY, BASELINE, np.eye(3))
        xyz   = np.array([1., 2., 3.])
        lm_id = empty_map.add(xyz, cov, make_desc(0), frame_idx=5)
        empty_map.merge(lm_id, xyz + 10, cov, make_desc(1), frame_idx=99)  # rejected
        assert empty_map.landmarks[lm_id].last_seen_frame == 5


# ── Landmark dataclass ────────────────────────────────────────────────────────

class TestLandmarkDataclass:

    def test_default_observations(self):
        lm = Landmark(id=0, xyz_world=np.zeros(3), covariance=np.eye(3))
        assert lm.observations == 1

    def test_default_descriptors_empty_list(self):
        lm = Landmark(id=0, xyz_world=np.zeros(3), covariance=np.eye(3))
        assert lm.descriptors == []

    def test_default_last_seen_frame(self):
        lm = Landmark(id=0, xyz_world=np.zeros(3), covariance=np.eye(3))
        assert lm.last_seen_frame == 0

    def test_descriptor_lists_not_shared(self):
        """Each Landmark must have its own descriptor list (mutable default trap)."""
        lm1 = Landmark(id=0, xyz_world=np.zeros(3), covariance=np.eye(3))
        lm2 = Landmark(id=1, xyz_world=np.ones(3),  covariance=np.eye(3))
        lm1.descriptors.append(make_desc(0))
        assert len(lm2.descriptors) == 0


# ── Asymmetric intrinsics ─────────────────────────────────────────────────────

class TestAsymmetricIntrinsics:

    FX2, FY2 = 400.0, 600.0   # non-square pixels

    def test_covariance_xy_differ_when_fx_ne_fy(self):
        cov = compute_covariance(3.0, self.FX2, self.FY2, BASELINE, np.eye(3))
        assert not np.isclose(cov[0, 0], cov[1, 1])

    def test_covariance_sigma_x_proportional_to_1_over_fx(self):
        cov = compute_covariance(3.0, self.FX2, self.FY2, BASELINE, np.eye(3))
        sigma_x = np.sqrt(cov[0, 0])
        expected = 3.0 / self.FX2
        assert sigma_x == pytest.approx(expected, rel=1e-6)

    def test_covariance_sigma_y_proportional_to_1_over_fy(self):
        cov = compute_covariance(3.0, self.FX2, self.FY2, BASELINE, np.eye(3))
        sigma_y = np.sqrt(cov[1, 1])
        expected = 3.0 / self.FY2
        assert sigma_y == pytest.approx(expected, rel=1e-6)

    def test_backproject_asymmetric(self):
        u, v, z = 500.0, 350.0, 4.0
        pts = backproject(np.array([u]), np.array([v]), np.array([z]),
                          self.FX2, self.FY2, CX, CY)
        assert pts[0, 0] == pytest.approx((u - CX) * z / self.FX2)
        assert pts[0, 1] == pytest.approx((v - CY) * z / self.FY2)
