import cv2
import numpy as np

from config.config import Config
from pipeline.base_pipeline import PipelineBase
from modules.landmarks.landmark_map import LandmarkMap, compute_covariance, backproject


class LandmarkPipeline(PipelineBase):
    """
    Visual odometry pipeline using SIFT landmarks + PnP pose estimation.

    Map lifecycle:
        - New keypoints with valid depth → add to LandmarkMap
        - Matched keypoints (ratio test) → merge (information-filter position update)
        - Pose from cv2.solvePnPRansac using 3D-2D correspondences

    Trajectory file format is identical to FlowPipeline for compatibility
    with render_images / compose_movie.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        self.sift = cv2.SIFT_create(nfeatures=self.cfg.max_keypoints)
        self.map  = LandmarkMap(
            match_ratio=0.75,
            max_merge_dist_3d=0.5,
        )

        K = self.cam_params.get_intrinsics()
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.baseline = float(self.params.get_baseline())
        self.K = K.astype(np.float64)

    # ------------------------------------------------------------------

    def compute_trajectory(self):
        print("Computing trajectory (landmark pipeline)...")

        # R_world_to_cam, t_world_to_cam for previous frame (PnP convention)
        R_prev = np.eye(3)
        t_prev = np.zeros(3)
        first_frame = True

        with open(self.traj_txt_path, "w") as traj_file:
            traj_file.write("frame, translation, rotation_matrix_flat\n")

            for i, left_rel in enumerate(self.left_files):
                img_left, img_right = self._load_stereo(left_rel)
                if img_left is None or img_right is None:
                    continue

                # ── Depth ──────────────────────────────────────────────
                disp  = self.disparity_solver.compute_disparity(img_left, img_right)
                depth = self.depth_solver.compute_depth(disp)

                # ── SIFT keypoints ──────────────────────────────────────
                kps, descs = self.sift.detectAndCompute(img_left, None)
                if descs is None or len(kps) == 0:
                    traj_file.write(f"{i}, {[0.,0.,0.]}, {np.eye(3).flatten().tolist()}\n")
                    continue

                kp_uv = np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float32)

                # ── Sample depth at keypoints ───────────────────────────
                us = np.clip(kp_uv[:, 0].astype(int), 0, depth.shape[1] - 1)
                vs = np.clip(kp_uv[:, 1].astype(int), 0, depth.shape[0] - 1)
                z_vals = depth[vs, us]
                valid  = (z_vals >= self.cfg.min_depth) & (z_vals <= self.cfg.max_depth) & (z_vals > 0)

                # ── First frame: initialise map, write identity ─────────
                if first_frame:
                    self._add_new_landmarks(kp_uv, descs, z_vals, valid,
                                            R_prev, t_prev, frame_idx=i)
                    first_frame = False
                    traj_file.write(f"{i}, {[0.,0.,0.]}, {np.eye(3).flatten().tolist()}\n")
                    continue

                # ── Match against map → PnP ─────────────────────────────
                pts_3d, pts_2d, lm_ids, kp_indices = self.map.get_correspondences(kp_uv, descs)

                R_curr, t_curr, inlier_mask = self._solve_pnp(pts_3d, pts_2d)

                if R_curr is None:
                    # PnP failed — write identity relative transform, keep prev pose
                    traj_file.write(f"{i}, {[0.,0.,0.]}, {np.eye(3).flatten().tolist()}\n")
                    # Still update map with new landmarks at prev pose
                    self._add_new_landmarks(kp_uv, descs, z_vals, valid,
                                            R_prev, t_prev, frame_idx=i,
                                            matched_kp_indices=set())
                    continue

                # ── Merge inlier landmarks ──────────────────────────────
                inliers = set()
                if inlier_mask is not None:
                    for j, lm_id in enumerate(lm_ids):
                        if inlier_mask[j]:
                            inliers.add(int(kp_indices[j]))
                            xyz_obs, cov_obs = self._obs_3d(
                                kp_uv[kp_indices[j]], z_vals[kp_indices[j]],
                                R_curr, t_curr
                            )
                            if xyz_obs is not None:
                                self.map.merge(lm_id, xyz_obs, cov_obs, descs[kp_indices[j]], i)

                # ── Add unmatched keypoints as new landmarks ────────────
                matched_kp = set(int(k) for k in kp_indices)
                self._add_new_landmarks(kp_uv, descs, z_vals, valid,
                                        R_curr, t_curr, frame_idx=i,
                                        matched_kp_indices=matched_kp)

                # ── Relative transform (for trajectory file) ───────────
                R_rel = R_curr @ R_prev.T
                t_rel = t_curr - R_rel @ t_prev
                traj_file.write(f"{i}, {t_rel.tolist()}, {R_rel.flatten().tolist()}\n")

                R_prev, t_prev = R_curr.copy(), t_curr.copy()

                if i % 20 == 0:
                    print(f"Processed {i} / {len(self.left_files)}  "
                          f"| map size: {self.map.size}  "
                          f"| correspondences: {len(pts_3d)}", flush=True)

        print("Trajectory computation complete. Data written to:", self.traj_txt_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _solve_pnp(self, pts_3d, pts_2d):
        """
        Run solvePnPRansac. Returns (R, t, inlier_mask) or (None, None, None).
        R, t follow the OpenCV convention: x_cam = R @ x_world + t.
        """
        if len(pts_3d) < 6:
            return None, None, None

        dist_coeffs = np.zeros(4)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d.astype(np.float64),
            pts_2d.astype(np.float64),
            self.K,
            dist_coeffs,
            reprojectionError=4.0,
            confidence=0.99,
            iterationsCount=200,
        )

        if not success or inliers is None or len(inliers) < 6:
            return None, None, None

        R, _ = cv2.Rodrigues(rvec)
        inlier_mask = np.zeros(len(pts_3d), dtype=bool)
        inlier_mask[inliers.flatten()] = True
        return R, tvec.flatten(), inlier_mask

    def _obs_3d(self, kp_uv, z, R_world_to_cam, t_world_to_cam):
        """
        Back-project a keypoint to world 3D + compute covariance.
        Returns (xyz_world, covariance) or (None, None) if depth invalid.
        """
        if z <= 0 or z > self.cfg.max_depth:
            return None, None

        xyz_cam = backproject(
            np.array([kp_uv[0]]), np.array([kp_uv[1]]), np.array([z]),
            self.fx, self.fy, self.cx, self.cy
        )[0]

        R_cam_to_world = R_world_to_cam.T
        t_cam_in_world = -R_cam_to_world @ t_world_to_cam
        xyz_world = R_cam_to_world @ xyz_cam + t_cam_in_world

        cov = compute_covariance(z, self.fx, self.fy, self.baseline, R_cam_to_world)
        return xyz_world, cov

    def _add_new_landmarks(self, kp_uv, descs, z_vals, valid_mask,
                           R_world_to_cam, t_world_to_cam,
                           frame_idx, matched_kp_indices=None):
        """Add valid unmatched keypoints as new landmarks."""
        if matched_kp_indices is None:
            matched_kp_indices = set()

        for idx in range(len(kp_uv)):
            if idx in matched_kp_indices:
                continue
            if not valid_mask[idx]:
                continue
            xyz_world, cov = self._obs_3d(
                kp_uv[idx], z_vals[idx], R_world_to_cam, t_world_to_cam
            )
            if xyz_world is not None:
                self.map.add(xyz_world, cov, descs[idx], frame_idx)
