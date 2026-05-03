import numpy as np
from scipy.spatial import cKDTree
from typing import Optional, Tuple

from modules.landmarks.landmark_map import LandmarkMap


class GuidedLandmarkMap(LandmarkMap):
    """
    LandmarkMap with geometry-guided matching.

    Instead of searching by descriptor (FLANN) and verifying geometry,
    we project each landmark into the current frame and search for the
    nearest detected keypoint within a pixel radius.  Descriptor distance
    is used only as an optional sanity check, not as the primary criterion.

    This fixes track fragmentation caused by descriptor drift: as the
    camera viewpoint changes, SIFT descriptors diverge and ratio-test
    matching fails even when the same physical point is clearly visible
    at a nearby pixel.  Spatial proximity is a more stable cue for
    consecutive-frame association.
    """

    def get_correspondences_guided(
        self,
        keypoints_2d: np.ndarray,
        descriptors: np.ndarray,
        visible: list,
        search_radius: float = 15.0,
        max_desc_dist: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Geometry-guided matching: project → nearest keypoint → optional descriptor check.

        Args:
            keypoints_2d:  (N, 2) detected keypoint pixel positions in current frame.
            descriptors:   (N, 128) SIFT descriptors.
            visible:       list of (lm_id, u_proj, v_proj) from frustum cull —
                           projected image position of each candidate landmark.
            search_radius: maximum pixel distance between projected landmark and
                           accepted keypoint.
            max_desc_dist: if set, reject matches whose L2 descriptor distance
                           exceeds this threshold (loose sanity check).

        Returns:
            pts_3d, pts_2d, lm_ids, kp_indices  — same contract as get_correspondences.
        """
        if len(keypoints_2d) == 0 or not visible:
            empty = np.empty((0,), dtype=int)
            return np.empty((0, 3)), np.empty((0, 2)), empty, empty

        # Build spatial index on new keypoints (N is small, ~320)
        kp_tree = cKDTree(keypoints_2d)

        proj_uvs        = np.array([[u, v] for _, u, v in visible], dtype=np.float32)
        lm_ids_visible  = [lm_id for lm_id, _, _ in visible]

        # For each projected landmark find the nearest keypoint within radius
        dists, kp_idxs = kp_tree.query(proj_uvs, k=1,
                                        distance_upper_bound=search_radius,
                                        workers=1)

        N = len(keypoints_2d)

        # Deduplicate: multiple landmarks may project near the same keypoint.
        # Keep the landmark whose projection is spatially closest.
        # seen_kp: kp_idx → (output_list_position, dist)
        pts_3d, pts_2d, lm_ids_out, kp_indices = [], [], [], []
        seen_kp: dict[int, tuple[int, float]] = {}

        for lm_id, dist, kp_idx in zip(lm_ids_visible, dists, kp_idxs):
            if kp_idx >= N:        # cKDTree returns N when no neighbour found
                continue

            if max_desc_dist is not None:
                desc_dist = float(np.linalg.norm(
                    descriptors[kp_idx].astype(np.float32) -
                    self._lm_to_desc[lm_id].astype(np.float32)
                ))
                if desc_dist > max_desc_dist:
                    continue

            if kp_idx in seen_kp:
                existing_pos, existing_dist = seen_kp[kp_idx]
                if existing_dist <= dist:
                    continue
                # Closer projection wins — replace in-place
                pts_3d[existing_pos]      = self.landmarks[lm_id].xyz_world
                pts_2d[existing_pos]      = keypoints_2d[kp_idx]
                lm_ids_out[existing_pos]  = lm_id
                kp_indices[existing_pos]  = kp_idx
                seen_kp[kp_idx]           = (existing_pos, dist)
            else:
                pos = len(pts_3d)
                pts_3d.append(self.landmarks[lm_id].xyz_world)
                pts_2d.append(keypoints_2d[kp_idx])
                lm_ids_out.append(lm_id)
                kp_indices.append(kp_idx)
                seen_kp[kp_idx] = (pos, dist)

        if not pts_3d:
            empty = np.empty((0,), dtype=int)
            return np.empty((0, 3)), np.empty((0, 2)), empty, empty

        return (np.array(pts_3d),
                np.array(pts_2d),
                np.array(lm_ids_out, dtype=int),
                np.array(kp_indices, dtype=int))
