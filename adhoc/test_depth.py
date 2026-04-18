
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
from modules.stereo.stereo_depth import StereoDepth
from modules.stereo.stereo_params_YAML import StereoParamsYAML
from modules.stereo.stereo_rectification import StereoRectification
from modules.stereo.stereo_disparity_RAFT import DisparityRAFT
from pipeline.visualization.video_composition import make_stacked_video


# ── Script-level debug knobs ───────────────────────────────────────────────────
class Solver(Enum):
    RAFT = auto()
    AANET = auto()

disparity_type = Solver.RAFT
single_frame   = False   # True = show one frame interactively; False = batch mode
img_idx        = 960     # frame index used in single-frame mode
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Depth visualisation adhoc test")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    # Resolve stereo checkpoint (relative → absolute, same as pipeline)
    stereo_ckpt = cfg.stereo_checkpoint if os.path.isabs(cfg.stereo_checkpoint) \
                  else os.path.join(project_root, cfg.stereo_checkpoint)

    # Init modules
    params        = StereoParamsYAML(cfg.yaml_file)
    rectification = StereoRectification(params)

    if disparity_type is Solver.RAFT:
        disparity_solver = DisparityRAFT(stereo_ckpt, rectification, cfg.raft_iters, cfg.raft_disparity_warmstart)
    elif disparity_type is Solver.AANET:
        from modules.stereo.stereo_disparity_AANET import DisparityAANet
        aanet_ckpt = os.path.join(project_root, "models/aanet/aanet_sceneflow-5aa5a24e.pth")
        disparity_solver = DisparityAANet(aanet_ckpt, rectification)

    depth_solver = StereoDepth(params)
    rectification_mask, _, __, ___ = rectification.get_rectification_masks()

    # ── Single frame mode ──────────────────────────────────────────────────────
    if single_frame:
        left_file  = cfg.dataset_path + f"img/image_0_{img_idx}.png"
        right_file = cfg.dataset_path + f"img/image_1_{img_idx}.png"

        img_left  = cv2.imread(left_file,  cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(right_file, cv2.IMREAD_GRAYSCALE)

        disparity = disparity_solver.compute_disparity(img_left, img_right)
        depth     = depth_solver.compute_depth(disparity)
        depth     = np.clip(depth, cfg.min_depth, cfg.max_depth)

        disparity_masked = disparity.copy()
        depth_masked     = depth.copy()
        disparity_masked[~rectification_mask] = np.nan
        depth_masked[~rectification_mask]     = np.nan

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].imshow(img_left,  cmap="gray");  axs[0, 0].set_title("Left Image")
        axs[0, 1].imshow(img_right, cmap="gray");  axs[0, 1].set_title("Right Image")

        im_disp  = axs[1, 0].imshow(-disparity_masked, cmap="jet")
        axs[1, 0].set_title("Disparity Map (Masked)"); axs[1, 0].axis("off")
        fig.colorbar(im_disp, ax=axs[1, 0], fraction=0.046, pad=0.04)

        im_depth = axs[1, 1].imshow(depth_masked, cmap="inferno")
        axs[1, 1].set_title("Metric Depth Map (Clipped & Masked)"); axs[1, 1].axis("off")
        fig.colorbar(im_depth, ax=axs[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
        return

    # ── Batch mode ─────────────────────────────────────────────────────────────
    left_txt  = cfg.dataset_path + "left_images.txt"
    right_txt = cfg.dataset_path + "right_images.txt"

    df_left  = pd.read_csv(left_txt,  sep=r'\s+', comment="#", names=["id", "timestamp", "image_name"])
    df_right = pd.read_csv(right_txt, sep=r'\s+', comment="#", names=["id", "timestamp", "image_name"])

    left_files  = df_left["image_name"].tolist()
    right_files = df_right["image_name"].tolist()
    if cfg.limit:
        left_files  = left_files[:cfg.limit]
        right_files = right_files[:cfg.limit]

    out_disp_dir  = os.path.join(cfg.output_path, "out_disp")
    out_depth_dir = os.path.join(cfg.output_path, "out_depth")

    if cfg.render_images:
        print("Rendering images.")
        os.makedirs(out_disp_dir,  exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)

        for i, (left_img, right_img) in enumerate(zip(left_files, right_files)):
            img_left  = cv2.imread(cfg.dataset_path + left_img,  cv2.IMREAD_GRAYSCALE)
            img_right = cv2.imread(cfg.dataset_path + right_img, cv2.IMREAD_GRAYSCALE)

            disparity = disparity_solver.compute_disparity(img_left, img_right)
            depth     = depth_solver.compute_depth(disparity)
            depth     = np.clip(depth, cfg.min_depth, cfg.max_depth)

            disparity_saved = disparity.copy()
            depth_saved     = depth.copy()
            disparity_saved[~rectification_mask] = 0
            depth_saved[~rectification_mask]     = 0

            index = int(left_img.split("_")[-1].split(".")[0])
            plt.imsave(os.path.join(out_disp_dir,  f"{index}_disparity.png"), -disparity_saved, cmap="jet")
            plt.imsave(os.path.join(out_depth_dir, f"{index}_depth.png"),      depth_saved,     cmap="inferno")

            if i % 20 == 0:
                print(f"Processed {i} of {len(left_files)} images.")

    if cfg.compose_movie:
        transformations = [
            lambda x: x,
            lambda x: x.replace("_0.png", "_1.png"),
            lambda x: os.path.join("out_disp",  f"{int(x.split('_')[-1].split('.')[0])}_disparity.png"),
            lambda x: os.path.join("out_depth", f"{int(x.split('_')[-1].split('.')[0])}_depth.png"),
        ]
        make_stacked_video(cfg.dataset_path, left_files,
                           os.path.join(cfg.output_path, "depth_video.mp4"),
                           transformations, 25, (2, 2))

    print("Processing complete.")


if __name__ == "__main__":
    main()
