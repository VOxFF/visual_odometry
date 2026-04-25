
import os
import sys
import argparse
from enum import Enum, auto

# ── sys.path setup (mirrors pipeline/pipeline.py) ─────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def _setup_external_paths(base_dir: str):
    raft_stereo_path = os.path.join(base_dir, "external", "RAFT-Stereo")
    raft_flow_path   = os.path.join(base_dir, "external", "RAFT-Flow")
    core_path        = os.path.join(raft_flow_path, "flow_core")
    aanet_path       = os.path.join(base_dir, "external", "aanet")
    for p in [raft_stereo_path, raft_flow_path, aanet_path]:
        if p not in sys.path:
            sys.path.append(p)
    if core_path not in sys.path:
        sys.path.insert(0, core_path)

_setup_external_paths(project_root)
# ──────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.config import Config
from modules.stereo.stereo_interfaces import StereoParamsInterface
from modules.stereo.stereo_depth import StereoDepth
from modules.stereo.stereo_params_YAML import StereoParamsYAML
from modules.stereo.stereo_rectification import StereoRectification
from modules.stereo.stereo_disparity_RAFT import DisparityRAFT
from modules.flow.flow_map_RAFT import OpticalFlowRAFT
from modules.keypoints.keypoints_uniform import UniformKeyPoints
from modules.keypoints.keypoints_shi_tomasi import ShiTomasiKeyPoints
from modules.keypoints.keypoints_3d import Keypoints3DXform
from modules.keypoints.keypoints_3d_flow import Keypoints3DFlow
from pipeline.visualization.video_composition import make_stacked_video


# ── Script-level debug knobs ───────────────────────────────────────────────────
class Solver(Enum):
    RAFT  = auto()
    AANET = auto()

disparity_type = Solver.RAFT
single_frame   = False
img_idx        = 960     # frame index for single-frame mode
z_labels       = False

# Multi-frame knobs
render_images  = True
compose_movie  = True
limit          = 0
k              = 25      # re-initialise keypoints every k frames

# Depth colour range for flow arrows
min_dist = 0.1   # metres
max_dist = 6.0   # metres
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Keypoints flow visualisation adhoc test")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    stereo_ckpt = cfg.stereo_checkpoint if os.path.isabs(cfg.stereo_checkpoint) \
                  else os.path.join(project_root, cfg.stereo_checkpoint)
    flow_ckpt   = cfg.flow_checkpoint if os.path.isabs(cfg.flow_checkpoint) \
                  else os.path.join(project_root, cfg.flow_checkpoint)

    params        = StereoParamsYAML(cfg.yaml_file)
    rectification = StereoRectification(params)

    if disparity_type is Solver.RAFT:
        disparity_solver = DisparityRAFT(stereo_ckpt, rectification, cfg.raft_iters, cfg.raft_disparity_warmstart)
    elif disparity_type is Solver.AANET:
        from modules.stereo.stereo_disparity_AANET import DisparityAANet
        aanet_ckpt = os.path.join(project_root, "models/aanet/aanet_sceneflow-5aa5a24e.pth")
        disparity_solver = DisparityAANet(aanet_ckpt, rectification)

    depth_solver = StereoDepth(params)
    flow_solver  = OpticalFlowRAFT(flow_ckpt, rectification, cfg.raft_iters, cfg.raft_optflow_warmstart)

    stereo_mask, _, __, ___ = rectification.get_rectification_masks()
    cam_params = params.get_camera_params(StereoParamsInterface.StereoCamera.LEFT)

    if cfg.keypoints_detector == 'shi_tomasi':
        pts_src = ShiTomasiKeyPoints(stereo_mask)
    else:
        pts_src = UniformKeyPoints(stereo_mask)

    pts_xform = Keypoints3DXform(cam_params, cfg.subpixel_keypoints)
    pts_flow  = Keypoints3DFlow(cam_params, pts_xform, stereo_mask, cfg.subpixel_keypoints)

    print(f"Max detectable z = {params.get_z_max():.2f} m")

    # ── Single frame mode ──────────────────────────────────────────────────────
    if single_frame:
        f1_left  = os.path.join(cfg.dataset_path, f"img/image_0_{img_idx}.png")
        f2_left  = os.path.join(cfg.dataset_path, f"img/image_0_{img_idx + 1}.png")
        f1_right = os.path.join(cfg.dataset_path, f"img/image_1_{img_idx}.png")
        f2_right = os.path.join(cfg.dataset_path, f"img/image_1_{img_idx + 1}.png")

        img1_left  = cv2.imread(f1_left,  cv2.IMREAD_GRAYSCALE)
        img2_left  = cv2.imread(f2_left,  cv2.IMREAD_GRAYSCALE)
        img1_right = cv2.imread(f1_right, cv2.IMREAD_GRAYSCALE)

        if img1_left is None or img2_left is None or img1_right is None:
            raise ValueError("One or more images not found. Check dataset_path in config.")

        disparity1 = disparity_solver.compute_disparity(img1_left, img1_right)
        depth1     = depth_solver.compute_depth(disparity1)

        disparity2 = disparity_solver.compute_disparity(img2_left, cv2.imread(f2_right, cv2.IMREAD_GRAYSCALE))
        depth2     = depth_solver.compute_depth(disparity2)

        flow_uv   = flow_solver.compute_flow(img1_left, img2_left)
        keypoints = pts_src.get_keypoints(img1_left, max_number=cfg.max_keypoints)

        keypoints_3d_f1              = pts_xform.to_3d(keypoints, depth1)
        keypoints_3d_f2, valid_mask  = pts_flow.compute_3d_flow(keypoints, depth1, depth2, flow_uv)
        projected_f2                 = pts_xform.to_2d(keypoints_3d_f2[valid_mask])

        keypoints_valid      = keypoints[valid_mask]
        keypoints_3d_valid_f1 = keypoints_3d_f1[valid_mask]
        keypoints_3d_valid_f2 = keypoints_3d_f2[valid_mask]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(img1_left, cmap="gray"); ax1.set_title(f"Frame {img_idx}")
        ax2.imshow(img2_left, cmap="gray"); ax2.set_title(f"Frame {img_idx + 1}")

        for i in range(len(keypoints_valid)):
            x1, y1 = keypoints_valid[i]
            x2, y2 = projected_f2[i]
            depth_val = keypoints_3d_valid_f1[i, 2]
            dz        = keypoints_3d_valid_f2[i, 2] - depth_val

            ax1.plot(x1, y1, "ro", markersize=4)
            ax2.plot(x2, y2, "rx", markersize=8)

            color = "yellow" if min_dist <= depth_val <= max_dist else "red"
            dx, dy = x2 - x1, y2 - y1
            ax1.arrow(x1, y1, dx, dy, head_width=2, head_length=3, color=color, linewidth=1)
            ax2.arrow(x1, y1, dx, dy, head_width=2, head_length=3, color=color, linewidth=1)

            if z_labels:
                text_color = "yellow" if abs(dz) < 1 else "red"
                ax1.text(x1 + 4, y1 + 4, f"{dz:.2f}", color=text_color, fontsize=8)

        for ax in (ax1, ax2):
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        return

    # ── Multi-frame mode ───────────────────────────────────────────────────────
    left_txt = os.path.join(cfg.dataset_path, "left_images.txt")
    df_left  = pd.read_csv(left_txt, sep=r'\s+', comment="#", names=["id", "timestamp", "image_name"])
    left_files = df_left["image_name"].tolist()
    if cfg.limit:
        left_files = left_files[:cfg.limit]

    out_dir = os.path.join(cfg.output_path, "out_kp_flow")

    if render_images:
        os.makedirs(out_dir, exist_ok=True)

        tracks_2D          = []
        track_start_z      = []
        current_2D         = None
        current_3D         = None

        for i in range(len(left_files) - 1):
            frame1_path = cfg.dataset_path + left_files[i]
            frame2_path = cfg.dataset_path + left_files[i + 1]
            right1_path = frame1_path.replace("image_0_", "image_1_")
            right2_path = frame2_path.replace("image_0_", "image_1_")

            img1_left  = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
            img2_left  = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
            img1_right = cv2.imread(right1_path, cv2.IMREAD_GRAYSCALE)
            img2_right = cv2.imread(right2_path, cv2.IMREAD_GRAYSCALE)
            if any(x is None for x in [img1_left, img2_left, img1_right, img2_right]):
                continue

            disp1  = disparity_solver.compute_disparity(img1_left, img1_right)
            depth1 = depth_solver.compute_depth(disp1)
            disp2  = disparity_solver.compute_disparity(img2_left, img2_right)
            depth2 = depth_solver.compute_depth(disp2)

            reinit = (i % k == 0) or (current_2D is None)

            if not reinit:
                flow_uv = flow_solver.compute_flow(img1_left, img2_left)
                new_3D, valid_mask = pts_flow.compute_3d_flow(current_2D, depth1, depth2, flow_uv)
                if new_3D.shape[0] < 4:
                    reinit = True

            if reinit:
                new_2D    = pts_src.get_keypoints(img1_left, max_number=cfg.max_keypoints)
                new_3D    = pts_xform.to_3d(new_2D, depth1)
                valid_3D  = new_3D[:, 2] > 0
                current_2D = new_2D[valid_3D]
                current_3D = new_3D[valid_3D]
                tracks_2D     = [[(p[0], p[1], q[2])] for p, q in zip(current_2D, current_3D)]
                track_start_z = [q[2] for q in current_3D]
            else:
                new_3D      = new_3D[valid_mask]
                new_2D      = pts_xform.to_2d(new_3D)
                current_3D  = new_3D
                current_2D  = new_2D

                new_tracks, new_start_z = [], []
                idx_v = 0
                for j, was_valid in enumerate(valid_mask):
                    if was_valid:
                        new_tracks.append(tracks_2D[j] + [(current_2D[idx_v][0], current_2D[idx_v][1], current_3D[idx_v, 2])])
                        new_start_z.append(track_start_z[j])
                        idx_v += 1
                tracks_2D, track_start_z = new_tracks, new_start_z

            vis = cv2.cvtColor(img1_left, cv2.COLOR_GRAY2BGR)
            for track, start_z in zip(tracks_2D, track_start_z):
                if len(track) < 2:
                    continue
                for s in range(len(track) - 1):
                    current_z = track[s + 1][2]
                    color = (0, 255, 0) if min_dist <= current_z <= max_dist else (255, 0, 0)
                    cv2.line(vis, (int(track[s][0]), int(track[s][1])),
                             (int(track[s + 1][0]), int(track[s + 1][1])), color, thickness=2)

            cv2.imwrite(os.path.join(out_dir, f"{i:06d}.png"), vis)
            if i % 20 == 0:
                print(f"Processed {i} / {len(left_files) - 1} frames")

    if compose_movie:
        print("Composing movie.")
        flow_files = left_files[:len(left_files) - 1]
        rel_out = os.path.relpath(cfg.output_path, cfg.dataset_path)
        transformations = [
            lambda x: os.path.join(rel_out, "out_kp_flow", f"{int(x.split('_')[-1].split('.')[0]):06d}.png"),
        ]
        make_stacked_video(cfg.dataset_path, flow_files,
                           os.path.join(cfg.output_path, "keypoints_video.mp4"),
                           transformations)

    print("Processing complete.")


if __name__ == "__main__":
    main()
