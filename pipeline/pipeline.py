import os
import sys
import re
import ast
import itertools
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.config import Config

# Add RAFT-Stereo and RAFT-Flow to sys.path
def _setup_external_paths(base_dir: str):
    raft_stereo_path = os.path.join(base_dir, "external", "RAFT-Stereo")
    raft_flow_path   = os.path.join(base_dir, "external", "RAFT-Flow")
    core_path        = os.path.join(raft_flow_path, "flow_core")
    for p in [raft_stereo_path, raft_flow_path]:
        if p not in sys.path:
            sys.path.append(p)
    if core_path not in sys.path:
        sys.path.insert(0, core_path)

_setup_external_paths(os.path.join(os.path.dirname(__file__), ".."))

from modules.stereo.stereo_interfaces import StereoParamsInterface
from modules.stereo.stereo_depth import StereoDepth
from modules.stereo.stereo_params_YAML import StereoParamsYAML
from modules.stereo.stereo_rectification import StereoRectification
from modules.stereo.stereo_disparity_RAFT import DisparityRAFT
from modules.flow.flow_map_RAFT import OpticalFlowRAFT
from modules.keypoints.keypoints_uniform import UniformKeyPoints
from modules.keypoints.keypoints_3d import Keypoints3DXform
from modules.keypoints.keypoints_3d_flow import Keypoints3DFlow
from modules.pose.camera_svd_xform import CameraRansacXform
from modules.imu.imu_YAML import YamlIMU
from modules.io.data_utils import match_ground_truth_positions
from pipeline.visualization.plot_3d import TrajectoryPlot
from pipeline.visualization.video_composition import make_stacked_video


class CameraTrackingPipeline:

    def __init__(self, config: Config):
        self.cfg = config

        # Resolve checkpoint paths relative to project root if not absolute
        base = os.path.dirname(os.path.dirname(__file__))
        stereo_ckpt = self.cfg.stereo_checkpoint if os.path.isabs(self.cfg.stereo_checkpoint) \
                      else os.path.join(base, self.cfg.stereo_checkpoint)
        flow_ckpt   = self.cfg.flow_checkpoint if os.path.isabs(self.cfg.flow_checkpoint) \
                      else os.path.join(base, self.cfg.flow_checkpoint)

        # Calibration + rectification
        self.params       = StereoParamsYAML(self.cfg.yaml_file)
        self.rectification = StereoRectification(self.params)

        # Solvers
        self.disparity_solver = DisparityRAFT(stereo_ckpt, self.rectification, self.cfg.raft_iters)
        self.depth_solver     = StereoDepth(self.params)
        self.flow_solver      = OpticalFlowRAFT(flow_ckpt, self.rectification, self.cfg.raft_iters)

        # Keypoints
        stereo_mask, _, _, _ = self.rectification.get_rectification_masks()
        cam_params = self.params.get_camera_params(StereoParamsInterface.StereoCamera.LEFT)
        self.pts_src   = UniformKeyPoints(stereo_mask)
        self.pts_xform = Keypoints3DXform(cam_params)
        self.pts_flow  = Keypoints3DFlow(cam_params, self.pts_xform, stereo_mask)

        # Pose estimator
        self.cam_estimator = CameraRansacXform()

        # Output paths
        os.makedirs(self.cfg.output_path, exist_ok=True)
        self.traj_txt_path = os.path.join(self.cfg.output_path, "camera_trajectory.txt")
        self.truth_txt_path = os.path.join(self.cfg.dataset_path, "groundtruth.txt")
        self.traj_img_dir  = os.path.join(self.cfg.output_path, "out_traj")
        os.makedirs(self.traj_img_dir, exist_ok=True)

        # Image list
        self.left_txt = os.path.join(self.cfg.dataset_path, "left_images.txt")
        df_left = pd.read_csv(self.left_txt, delim_whitespace=True, comment="#",
                              names=["id", "timestamp", "image_name"])
        self.left_files = df_left["image_name"].tolist()
        if self.cfg.limit:
            self.left_files = self.left_files[:self.cfg.limit]

    # ------------------------------------------------------------------

    def run(self):
        if self.cfg.compute_trajectory:
            self.compute_trajectory()
        if self.cfg.render_images:
            valid_indices = self.render_images()
        if self.cfg.compose_movie:
            self.compose_movie(valid_indices)
        print("Processing complete.")

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
                    disp1  = self.disparity_solver.compute_disparity(img_left1, img_right1)
                    depth1 = self.depth_solver.compute_depth(disp1)
                    disp2  = self.disparity_solver.compute_disparity(img_left2, img_right2)
                    depth2 = self.depth_solver.compute_depth(disp2)
                    flow_uv = self.flow_solver.compute_flow(img_left1, img_left2)
                    prev_img_left, prev_img_right, prev_depth = img_left2, img_right2, depth2
                else:
                    img_left1, depth1 = prev_img_left, prev_depth
                    img_left2, img_right2 = self._load_stereo(self.left_files[i + 1])
                    if img_left2 is None or img_right2 is None:
                        print(f"Skipping frame {i} due to missing images.")
                        continue
                    disp2  = self.disparity_solver.compute_disparity(img_left2, img_right2)
                    depth2 = self.depth_solver.compute_depth(disp2)
                    flow_uv = self.flow_solver.compute_flow(img_left1, img_left2)
                    prev_img_left, prev_img_right, prev_depth = img_left2, img_right2, depth2

                # 2D keypoints → filter by depth range
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
                dz_mask = np.abs(new_3D[:, 2] - old_3D[:, 2]) <= self.cfg.dz_threshold
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

    # ------------------------------------------------------------------

    def render_images(self) -> list:
        print("Rendering images...")

        # Initial camera-to-world rotation from IMU/calibration
        imu = YamlIMU(self.cfg.yaml_file, self.left_txt, self.truth_txt_path)
        R_cam_in_world = imu.get_initial_pose()

        # Accumulate trajectory from file
        T_global = np.eye(4)
        global_positions, global_Ts = [], []

        with open(self.traj_txt_path, 'r') as f:
            f.readline()  # skip header
            lines = itertools.islice(f, self.cfg.limit) if self.cfg.limit else f
            for line in lines:
                matches = re.findall(r'\[.*?\]', line)
                if len(matches) < 2:
                    continue
                t_rel = np.array(ast.literal_eval(matches[0]))
                R_rel = np.array(ast.literal_eval(matches[1])).reshape(3, 3)

                T_rel_inv = np.eye(4)
                T_rel_inv[:3, :3] = R_rel.T
                T_rel_inv[:3, 3]  = -R_rel.T @ t_rel
                T_global = T_global @ T_rel_inv

                world_pos = R_cam_in_world @ T_global[:3, 3]
                R_world   = R_cam_in_world @ T_global[:3, :3] @ R_cam_in_world.T
                T_world   = np.eye(4)
                T_world[:3, :3] = R_world
                T_world[:3, 3]  = world_pos

                global_positions.append(world_pos)
                global_Ts.append(T_world)

        # Match with GT
        combined = match_ground_truth_positions(global_positions, self.left_txt, self.truth_txt_path)
        valid_indices = [i for i, (_, gt) in enumerate(combined) if gt is not None]
        valid_pairs   = [combined[i] for i in valid_indices]
        valid_Ts      = [global_Ts[i] for i in valid_indices]
        print(f"Frames with GT match: {len(valid_pairs)} / {len(combined)}")

        if not valid_pairs:
            print("No frames with GT match found — check timestamps and tolerance.")
            return valid_indices

        # Origin-align both trajectories
        first_est = valid_pairs[0][0].copy()
        first_gt  = valid_pairs[0][1].copy()
        aligned_pairs = [(comp - first_est, gt - first_gt) for comp, gt in valid_pairs]
        for T in valid_Ts:
            T[:3, 3] -= first_est

        tp = TrajectoryPlot(aligned_pairs,
                            elevation=self.cfg.elevation,
                            azimuth=self.cfg.azimuth,
                            zoom_distance=self.cfg.zoom_distance,
                            small_font_size=8)

        print("Rendering trajectory images...")
        for idx, (current_T) in enumerate(valid_Ts):
            fig = tp.plot(current_T, idx)
            fig.savefig(os.path.join(self.traj_img_dir, f"traj_{idx:06d}.png"))
            plt.close(fig)
            if idx % 20 == 0:
                print(f"Rendered {idx + 1} / {len(valid_pairs)}")

        return valid_indices

    # ------------------------------------------------------------------

    def compose_movie(self, valid_indices: list):
        print("Composing movie...")
        valid_image_files = [self.left_files[i] for i in valid_indices]
        img_to_traj_idx   = {f: idx for idx, f in enumerate(valid_image_files)}
        transformations = [
            lambda x: x,
            lambda x: os.path.join(self.traj_img_dir, f"traj_{img_to_traj_idx[x]:06d}.png"),
        ]
        out_path = os.path.join(self.cfg.output_path, "cam_tracking_video.mp4")
        make_stacked_video(self.cfg.dataset_path, valid_image_files, out_path, transformations)
        print(f"Movie composed as {out_path}")

    # ------------------------------------------------------------------

    def _load_stereo(self, left_rel_path: str):
        left_path  = os.path.join(self.cfg.dataset_path, left_rel_path)
        right_path = left_path.replace("image_0_", "image_1_")
        img_left  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        return img_left, img_right
