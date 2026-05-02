import os
import sys
import re
from datetime import datetime
import ast
import itertools
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from config.config import Config

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
from modules.imu.imu_YAML import YamlIMU
from modules.io.data_utils import match_ground_truth_positions
from pipeline.visualization.plot_3d import TrajectoryPlot
from pipeline.visualization.video_composition import make_stacked_video


class PipelineBase(ABC):

    def __init__(self, config: Config):
        self.cfg = config

        base = os.path.dirname(os.path.dirname(__file__))
        stereo_ckpt = self.cfg.stereo_checkpoint if os.path.isabs(self.cfg.stereo_checkpoint) \
                      else os.path.join(base, self.cfg.stereo_checkpoint)

        # Calibration + rectification
        self.params        = StereoParamsYAML(self.cfg.yaml_file)
        self.rectification = StereoRectification(self.params)

        # Stereo solvers (shared by all pipelines)
        self.disparity_solver = DisparityRAFT(stereo_ckpt, self.rectification,
                                              self.cfg.raft_iters,
                                              self.cfg.raft_disparity_warmstart)
        self.depth_solver = StereoDepth(self.params)

        # Camera geometry
        self.stereo_mask, _, _, _ = self.rectification.get_rectification_masks()
        self.cam_params = self.params.get_camera_params(StereoParamsInterface.StereoCamera.LEFT)

        # Output paths
        os.makedirs(self.cfg.output_path, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.traj_txt_path  = os.path.join(self.cfg.output_path, f"camera_trajectory_{ts}.txt")
        self.truth_txt_path = os.path.join(self.cfg.dataset_path, "groundtruth.txt")
        self.traj_img_dir   = os.path.join(self.cfg.output_path, "out_traj")
        os.makedirs(self.traj_img_dir, exist_ok=True)

        # Image list
        self.left_txt = os.path.join(self.cfg.dataset_path, "left_images.txt")
        df_left = pd.read_csv(self.left_txt, delim_whitespace=True, comment="#",
                              names=["id", "timestamp", "image_name"])
        all_files = df_left["image_name"].tolist()
        self.frame_start = self.cfg.start_frame
        self.left_files  = all_files[self.frame_start:]
        if self.cfg.limit:
            self.left_files = self.left_files[:self.cfg.limit]

    # ------------------------------------------------------------------

    def run(self):
        valid_indices = None
        if self.cfg.compute_trajectory:
            self.compute_trajectory()
        if self.cfg.render_images:
            valid_indices = self.render_images()
        if self.cfg.compose_movie:
            self.compose_movie(valid_indices)
        print("Processing complete.")

    # ------------------------------------------------------------------

    @abstractmethod
    def compute_trajectory(self):
        pass

    # ------------------------------------------------------------------

    def render_images(self) -> list:
        print("Rendering images...")

        imu = YamlIMU(self.cfg.yaml_file, self.left_txt, self.truth_txt_path)
        R_cam_in_world = imu.get_initial_pose()

        T_global = np.eye(4)
        global_positions, global_Ts = [], []

        with open(self.traj_txt_path, 'r') as f:
            f.readline()
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

        combined     = match_ground_truth_positions(global_positions, self.left_txt, self.truth_txt_path,
                                                    start_index=self.frame_start)
        valid_indices = [i for i, (_, gt) in enumerate(combined) if gt is not None]
        valid_pairs   = [combined[i] for i in valid_indices]
        valid_Ts      = [global_Ts[i] for i in valid_indices]
        print(f"Frames with GT match: {len(valid_pairs)} / {len(combined)}")

        if not valid_pairs:
            print("No frames with GT match found — check timestamps and tolerance.")
            return valid_indices

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
        for idx, current_T in enumerate(valid_Ts):
            fig = tp.plot(current_T, idx)
            fig.savefig(os.path.join(self.traj_img_dir, f"traj_{idx:06d}.png"))
            plt.close(fig)
            if idx % 20 == 0:
                print(f"Rendered {idx + 1} / {len(valid_pairs)}")

        return valid_indices

    # ------------------------------------------------------------------

    def compose_movie(self, valid_indices: list):
        print("Composing movie...")
        if not valid_indices:
            print("No valid frames to compose — skipping movie.")
            return
        valid_image_files = [self.left_files[i] for i in valid_indices]
        img_to_traj_idx   = {f: idx for idx, f in enumerate(valid_image_files)}
        transformations = [
            lambda x: x,
            lambda x: os.path.join(self.traj_img_dir, f"traj_{img_to_traj_idx[x]:06d}.png"),
        ]
        ts       = os.path.splitext(os.path.basename(self.traj_txt_path))[0].split("_", 2)[2]
        out_path = os.path.join(self.cfg.output_path, f"cam_tracking_video_{ts}.mp4")
        make_stacked_video(self.cfg.dataset_path, valid_image_files, out_path, transformations)
        print(f"Movie composed as {out_path}")

    # ------------------------------------------------------------------

    def _load_stereo(self, left_rel_path: str):
        left_path  = os.path.join(self.cfg.dataset_path, left_rel_path)
        right_path = left_path.replace("image_0_", "image_1_")
        img_left  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        return img_left, img_right
