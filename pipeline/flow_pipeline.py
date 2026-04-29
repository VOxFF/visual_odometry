import numpy as np

from config.config import Config
from pipeline.base_pipeline import PipelineBase
from modules.flow.flow_map_RAFT import OpticalFlowRAFT
from modules.keypoints.keypoints_uniform import UniformKeyPoints
from modules.keypoints.keypoints_shi_tomasi import ShiTomasiKeyPoints
from modules.keypoints.keypoints_3d import Keypoints3DXform
from modules.keypoints.keypoints_3d_flow import Keypoints3DFlow
from modules.pose.camera_svd_xform import CameraRansacXform

import os


class FlowPipeline(PipelineBase):
    """
    Visual odometry pipeline using RAFT optical flow + SVD/RANSAC pose estimation.
    Tracks a uniform or Shi-Tomasi keypoint grid frame-to-frame via dense optical flow.
    """

    def __init__(self, config: Config):
        super().__init__(config)

        base = os.path.dirname(os.path.dirname(__file__))
        flow_ckpt = self.cfg.flow_checkpoint if os.path.isabs(self.cfg.flow_checkpoint) \
                    else os.path.join(base, self.cfg.flow_checkpoint)

        self.flow_solver = OpticalFlowRAFT(flow_ckpt, self.rectification,
                                           self.cfg.raft_iters,
                                           self.cfg.raft_optflow_warmstart)

        if self.cfg.keypoints_detector == 'shi_tomasi':
            self.pts_src = ShiTomasiKeyPoints(self.stereo_mask)
        else:
            self.pts_src = UniformKeyPoints(self.stereo_mask)

        self.pts_xform = Keypoints3DXform(self.cam_params, self.cfg.subpixel_keypoints)
        self.pts_flow  = Keypoints3DFlow(self.cam_params, self.pts_xform,
                                         self.stereo_mask, self.cfg.subpixel_keypoints)
        self.cam_estimator = CameraRansacXform()

    # ------------------------------------------------------------------

    def compute_trajectory(self):
        print("Computing trajectory over frames...")
        T_global = np.eye(4)
        prev_img_left = prev_img_right = prev_depth = None

        with open(self.traj_txt_path, "w") as traj_file:
            traj_file.write("frame, translation, rotation_matrix_flat\n")

            for i in range(len(self.left_files) - 1):
                if prev_depth is None:
                    img_left1, img_right1 = self._load_stereo(self.left_files[i])
                    img_left2, img_right2 = self._load_stereo(self.left_files[i + 1])
                    if any(x is None for x in [img_left1, img_right1, img_left2, img_right2]):
                        print(f"Skipping frame {i} due to missing images.")
                        continue
                    disp1   = self.disparity_solver.compute_disparity(img_left1, img_right1)
                    depth1  = self.depth_solver.compute_depth(disp1)
                    disp2   = self.disparity_solver.compute_disparity(img_left2, img_right2)
                    depth2  = self.depth_solver.compute_depth(disp2)
                    flow_uv = self.flow_solver.compute_flow(img_left1, img_left2)
                    prev_img_left, prev_img_right, prev_depth = img_left2, img_right2, depth2
                else:
                    img_left1, depth1 = prev_img_left, prev_depth
                    img_left2, img_right2 = self._load_stereo(self.left_files[i + 1])
                    if img_left2 is None or img_right2 is None:
                        print(f"Skipping frame {i} due to missing images.")
                        continue
                    disp2   = self.disparity_solver.compute_disparity(img_left2, img_right2)
                    depth2  = self.depth_solver.compute_depth(disp2)
                    flow_uv = self.flow_solver.compute_flow(img_left1, img_left2)
                    prev_img_left, prev_img_right, prev_depth = img_left2, img_right2, depth2

                kp2d = self.pts_src.get_keypoints(img_left1, max_number=self.cfg.max_keypoints)
                kp3d = self.pts_xform.to_3d(kp2d, depth1)
                depth_mask = (kp3d[:, 2] >= self.cfg.min_depth) & (kp3d[:, 2] <= self.cfg.max_depth)
                kp3d, kp2d = kp3d[depth_mask], kp2d[depth_mask]

                kp3d_2, valid_mask = self.pts_flow.compute_3d_flow(kp2d, depth1, depth2, flow_uv)
                if kp3d_2.shape[0] < 4:
                    print("Too few valid keypoints; skipping.")
                    continue

                old_3D = kp3d[valid_mask]
                new_3D = kp3d_2[valid_mask]

                if self.cfg.log_dz_threshold:
                    dz_allowed = self.cfg.dz_threshold * np.log1p(old_3D[:, 2])
                else:
                    dz_allowed = self.cfg.dz_threshold
                dz_mask = np.abs(new_3D[:, 2] - old_3D[:, 2]) <= dz_allowed
                old_3D, new_3D = old_3D[dz_mask], new_3D[dz_mask]

                R_rel, t_rel = self.cam_estimator.compute_camera_xform(old_3D, new_3D)
                T_rel_inv = np.eye(4)
                T_rel_inv[:3, :3] = R_rel.T
                T_rel_inv[:3, 3]  = -R_rel.T @ t_rel
                T_global = T_global @ T_rel_inv

                traj_file.write(f"{i}, {t_rel.tolist()}, {R_rel.flatten().tolist()}\n")

                if i % 20 == 0:
                    print(f"Processed {i} / {len(self.left_files) - 1} frames.")

        print("Trajectory computation complete. Data written to:", self.traj_txt_path)
