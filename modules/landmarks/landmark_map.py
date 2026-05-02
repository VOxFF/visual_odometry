import cv2
import numpy as np
from typing import Optional, Tuple

from modules.landmarks.landmark import Landmark


def compute_covariance(z: float,
                       fx: float, fy: float,
                       baseline: float,
                       R_cam_to_world: np.ndarray,
                       sigma_pixel: float = 1.0,
                       sigma_disp: float = 1.0) -> np.ndarray:
    """
    Anisotropic Gaussian covariance for a stereo landmark at depth z.

    In camera frame:
        σ_x = z / fx * σ_pixel      (lateral, grows linearly with depth)
        σ_y = z / fy * σ_pixel
        σ_z = z² / (fx * baseline) * σ_disp   (depth, grows with z²)

    Rotated to world frame: Σ_world = R @ Σ_cam @ R.T
    """
    sx = (z / fx) * sigma_pixel
    sy = (z / fy) * sigma_pixel
    sz = (z ** 2 / (fx * baseline)) * sigma_disp
    cov_cam = np.diag([sx ** 2, sy ** 2, sz ** 2])
    return R_cam_to_world @ cov_cam @ R_cam_to_world.T


def backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray,
                fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Back-project pixel (u, v) at depth z to 3D camera-frame coordinates.

    Returns (N, 3) array.
    """
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return np.stack([X, Y, z], axis=1)


# FLANN k-d tree parameters for float32 L2 descriptors (SIFT).
# trees=5: number of parallel k-d trees — more trees = higher accuracy, slower build.
# checks=50: nodes visited per query — higher = more accurate, slower search.
_FLANN_INDEX_KDTREE = 1
_FLANN_INDEX_PARAMS  = dict(algorithm=_FLANN_INDEX_KDTREE, trees=5)
_FLANN_SEARCH_PARAMS = dict(checks=50)


class LandmarkMap:
    """
    Sparse 3D landmark map with SIFT descriptor matching.

    Each landmark stores its world-frame position, uncertainty covariance,
    and all SIFT descriptors from every observation (for future aging/voting).

    Matching index design
    ---------------------
    The index keeps exactly ONE descriptor per landmark — the latest merged
    descriptor. This bounds index size to map.size regardless of how many times
    each landmark has been observed, avoiding the BFMatcher ~262K row hard limit
    that would be hit after ~860 frames at 320 keypoints/frame.

    FLANN (FlannBasedMatcher) is used instead of BFMatcher. BFMatcher is O(N×M)
    exact search; FLANN builds a k-d tree for O(M × log N) approximate search —
    10-50× faster at large N with negligible accuracy loss for SIFT.

    Operations
    ----------
        get_correspondences  — match new-frame descriptors → 3D-2D pairs for PnP
        merge                — update existing landmark with new observation
        add                  — insert new landmark
    """

    def __init__(self,
                 match_ratio: float = 0.75,
                 max_merge_dist_3d: float = 0.5):
        """
        Args:
            match_ratio:       Lowe ratio-test threshold. A match passes only if
                               best_dist < match_ratio * second_best_dist.
                               Lower = stricter (fewer but more reliable matches).
            max_merge_dist_3d: Maximum Euclidean distance (metres) between a new
                               3D observation and an existing landmark for the merge
                               to be accepted. Guards against descriptor collisions
                               between geometrically distant points.
        """
        self.match_ratio       = match_ratio
        self.max_merge_dist_3d = max_merge_dist_3d

        self.landmarks: dict[int, Landmark] = {}
        self._next_id = 0

        # --- Matching index ---
        # _lm_to_desc: lm_id → latest descriptor (one entry per landmark).
        #   Updated on every merge so the index always reflects the most recent
        #   appearance of each landmark.
        # _lm_index_ids: ordered list of lm_ids matching rows of _desc_matrix.
        #   Rebuilt lazily together with _desc_matrix when _dirty=True.
        # _desc_matrix: (N, 128) float32 array passed to FLANN knnMatch.
        self._lm_to_desc: dict[int, np.ndarray] = {}
        self._lm_index_ids: list[int] = []
        self._desc_matrix: Optional[np.ndarray] = None
        self._dirty = False

        # FLANN approximate nearest-neighbour matcher for float32 L2 descriptors.
        self._matcher = cv2.FlannBasedMatcher(_FLANN_INDEX_PARAMS, _FLANN_SEARCH_PARAMS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.landmarks)

    def get_correspondences(self,
                            keypoints_2d: np.ndarray,
                            descriptors: np.ndarray,
                            visible_ids: Optional[list] = None,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Match new-frame descriptors against the landmark map.

        Uses Lowe's ratio test to filter ambiguous matches, then deduplicates
        so each landmark appears at most once (keeps the closest query match).

        Args:
            keypoints_2d: (N, 2) pixel coordinates in the new frame.
            descriptors:  (N, 128) SIFT descriptors.
            visible_ids:  Optional list of landmark ids to match against.
                          When provided (frustum culling), only these landmarks
                          are included in the search index.  This keeps the index
                          small even as the global map grows, reducing false
                          ratio-test passes and improving RANSAC inlier ratio.
                          When None the full map is searched.

        Returns:
            pts_3d     : (M, 3) matched landmark world positions
            pts_2d     : (M, 2) matched pixel positions
            lm_ids     : (M,)   landmark ids (for subsequent merge calls)
            kp_indices : (M,)   which query keypoint each match came from
        """
        if len(descriptors) == 0:
            empty = np.empty((0,), dtype=int)
            return np.empty((0, 3)), np.empty((0, 2)), empty, empty

        if visible_ids is not None:
            # Build a temporary matrix from the culled subset only.
            # Avoids searching the full (potentially 100K+) map and prevents
            # the ratio test from passing on geometrically impossible matches.
            index_ids = [lid for lid in visible_ids if lid in self._lm_to_desc]
            if len(index_ids) < 2:
                empty = np.empty((0,), dtype=int)
                return np.empty((0, 3)), np.empty((0, 2)), empty, empty
            train = np.array([self._lm_to_desc[lid] for lid in index_ids],
                             dtype=np.float32)
        else:
            if self.size < 2:
                # Need ≥ 2 landmarks for knnMatch k=2 ratio test
                empty = np.empty((0,), dtype=int)
                return np.empty((0, 3)), np.empty((0, 2)), empty, empty
            train, index_ids = self._get_matrix()
        matches = self._matcher.knnMatch(descriptors.astype(np.float32), train, k=2)

        pts_3d, pts_2d, lm_ids, kp_indices = [], [], [], []
        seen_lm: dict[int, tuple[int, float]] = {}  # lm_id → (output_idx, distance)

        for i, pair in enumerate(matches):
            if len(pair) < 2:
                continue
            m, n = pair

            # Lowe ratio test: accept only unambiguous matches
            if m.distance >= self.match_ratio * n.distance:
                continue

            lm_id = index_ids[m.trainIdx]

            if lm_id in seen_lm:
                existing_idx, existing_dist = seen_lm[lm_id]
                if existing_dist <= m.distance:
                    continue
                # Better match for same landmark: replace in-place (no duplicate row)
                pts_2d[existing_idx]     = keypoints_2d[i]
                kp_indices[existing_idx] = i
                seen_lm[lm_id]           = (existing_idx, m.distance)
            else:
                out_idx = len(pts_3d)
                pts_3d.append(self.landmarks[lm_id].xyz_world)
                pts_2d.append(keypoints_2d[i])
                lm_ids.append(lm_id)
                kp_indices.append(i)
                seen_lm[lm_id] = (out_idx, m.distance)

        if not pts_3d:
            empty = np.empty((0,), dtype=int)
            return np.empty((0, 3)), np.empty((0, 2)), empty, empty

        return (np.array(pts_3d),
                np.array(pts_2d),
                np.array(lm_ids, dtype=int),
                np.array(kp_indices, dtype=int))

    def merge(self,
              lm_id: int,
              xyz_world: np.ndarray,
              covariance: np.ndarray,
              descriptor: np.ndarray,
              frame_idx: int) -> bool:
        """
        Update an existing landmark with a new observation.

        Position is refined via information-filter (Kalman-style) fusion:
            Σ_new⁻¹  = Σ_old⁻¹ + Σ_obs⁻¹
            xyz_new  = Σ_new × (Σ_old⁻¹ × xyz_old + Σ_obs⁻¹ × xyz_obs)
        Each re-observation tightens the position estimate, weighted by certainty.

        The matching index entry for this landmark is updated to the latest
        descriptor so subsequent matches use the most recent appearance.

        Returns True if merged, False if the 3D consistency check failed
        (observation too far from existing landmark position).
        """
        lm = self.landmarks.get(lm_id)
        if lm is None:
            return False

        # 3D consistency guard: reject if new observation is geometrically far
        # from the stored position. Catches descriptor collisions (two different
        # physical points with similar SIFT descriptors).
        dist = float(np.linalg.norm(xyz_world - lm.xyz_world))
        if dist > self.max_merge_dist_3d:
            return False

        # Information-filter position update
        try:
            S_old_inv = np.linalg.inv(lm.covariance)
            S_obs_inv = np.linalg.inv(covariance)
            S_new     = np.linalg.inv(S_old_inv + S_obs_inv)
            xyz_new   = S_new @ (S_old_inv @ lm.xyz_world + S_obs_inv @ xyz_world)
        except np.linalg.LinAlgError:
            # Singular covariance (degenerate case) — fall back to simple average
            xyz_new = (lm.xyz_world + xyz_world) / 2.0
            S_new   = lm.covariance

        lm.xyz_world  = xyz_new
        lm.covariance = S_new
        lm.descriptors.append(descriptor)   # keep full history for future aging
        lm.observations += 1
        lm.last_seen_frame = frame_idx

        # Update matching index to latest descriptor (replaces old entry in-place,
        # index size stays constant at map.size)
        self._lm_to_desc[lm_id] = descriptor
        self._dirty = True
        return True

    def add(self,
            xyz_world: np.ndarray,
            covariance: np.ndarray,
            descriptor: np.ndarray,
            frame_idx: int) -> int:
        """
        Add a new landmark. Returns its id.
        """
        lm_id = self._next_id
        self._next_id += 1

        self.landmarks[lm_id] = Landmark(
            id=lm_id,
            xyz_world=xyz_world.copy(),
            covariance=covariance.copy(),
            descriptors=[descriptor],
            observations=1,
            last_seen_frame=frame_idx,
        )
        # Register in matching index
        self._lm_to_desc[lm_id] = descriptor
        self._lm_index_ids.append(lm_id)
        self._dirty = True
        return lm_id

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_matrix(self) -> Tuple[np.ndarray, list]:
        """
        Return (desc_matrix, lm_index_ids), rebuilding lazily if dirty.

        desc_matrix   : (N, 128) float32 — one row per landmark (latest descriptor)
        lm_index_ids  : list of lm_ids, parallel to desc_matrix rows
        """
        if self._dirty or self._desc_matrix is None:
            self._lm_index_ids = list(self._lm_to_desc.keys())
            self._desc_matrix  = np.array(
                [self._lm_to_desc[i] for i in self._lm_index_ids],
                dtype=np.float32
            )
            self._dirty = False
        return self._desc_matrix, self._lm_index_ids
