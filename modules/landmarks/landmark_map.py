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


class LandmarkMap:
    """
    Sparse 3D landmark map with SIFT descriptor matching.

    Each landmark stores its world-frame position, uncertainty covariance,
    and all SIFT descriptors from every observation.

    Operations:
        get_correspondences  — match new-frame descriptors → 3D-2D pairs for PnP
        merge                — update existing landmark with new observation
        add                  — insert new landmark
    """

    def __init__(self,
                 match_ratio: float = 0.75,
                 max_merge_dist_3d: float = 0.5):
        """
        Args:
            match_ratio:       Lowe ratio-test threshold (lower = stricter matching).
            max_merge_dist_3d: Max Euclidean distance (m) between new observation
                               and existing landmark to allow merge.
        """
        self.match_ratio      = match_ratio
        self.max_merge_dist_3d = max_merge_dist_3d

        self.landmarks: dict[int, Landmark] = {}
        self._next_id = 0

        # Flat descriptor index — grows as landmarks are added / merged
        self._desc_list: list[np.ndarray] = []   # (128,) float32 per entry
        self._lm_id_list: list[int]       = []   # landmark id parallel to _desc_list

        self._desc_matrix: Optional[np.ndarray] = None   # cached (N, 128)
        self._dirty = False

        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self.landmarks)

    def get_correspondences(self,
                            keypoints_2d: np.ndarray,
                            descriptors: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Match new-frame descriptors against the landmark map.

        Args:
            keypoints_2d: (N, 2) pixel coordinates in the new frame.
            descriptors:  (N, 128) SIFT descriptors.

        Returns:
            pts_3d     : (M, 3) matched landmark world positions
            pts_2d     : (M, 2) matched pixel positions
            lm_ids     : (M,)   landmark ids (for subsequent merge calls)
            kp_indices : (M,)   which query keypoint each match came from
        """
        if len(self._desc_list) < 2 or len(descriptors) == 0:
            empty = np.empty((0,), dtype=int)
            return np.empty((0, 3)), np.empty((0, 2)), empty, empty

        train = self._get_matrix()
        matches = self._matcher.knnMatch(descriptors.astype(np.float32), train, k=2)

        pts_3d, pts_2d, lm_ids, kp_indices = [], [], [], []
        seen_lm: dict[int, float] = {}   # deduplicate: keep best match per landmark

        for i, pair in enumerate(matches):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance >= self.match_ratio * n.distance:
                continue

            lm_id = self._lm_id_list[m.trainIdx]
            # Keep only the closest query match per landmark
            if lm_id in seen_lm and seen_lm[lm_id] <= m.distance:
                continue
            seen_lm[lm_id] = m.distance

            pts_3d.append(self.landmarks[lm_id].xyz_world)
            pts_2d.append(keypoints_2d[i])
            lm_ids.append(lm_id)
            kp_indices.append(i)

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

        Position is refined via information-filter fusion:
            Σ_new⁻¹  = Σ_old⁻¹ + Σ_obs⁻¹
            xyz_new  = Σ_new × (Σ_old⁻¹ × xyz_old + Σ_obs⁻¹ × xyz_obs)

        Returns True if merged, False if 3D consistency check failed.
        """
        lm = self.landmarks.get(lm_id)
        if lm is None:
            return False

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
            xyz_new = (lm.xyz_world + xyz_world) / 2.0   # fallback: average
            S_new   = lm.covariance

        lm.xyz_world  = xyz_new
        lm.covariance = S_new
        lm.descriptors.append(descriptor)
        lm.observations += 1
        lm.last_seen_frame = frame_idx

        self._desc_list.append(descriptor)
        self._lm_id_list.append(lm_id)
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
        self._desc_list.append(descriptor)
        self._lm_id_list.append(lm_id)
        self._dirty = True
        return lm_id

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_matrix(self) -> np.ndarray:
        if self._dirty or self._desc_matrix is None:
            self._desc_matrix = np.array(self._desc_list, dtype=np.float32)
            self._dirty = False
        return self._desc_matrix
