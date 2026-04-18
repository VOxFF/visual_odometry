
import os
import sys
import argparse

# ── sys.path setup (mirrors pipeline/pipeline.py) ─────────────────────────────
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def _setup_external_paths(base_dir: str):
    raft_stereo_path = os.path.join(base_dir, "external", "RAFT-Stereo")
    raft_flow_path   = os.path.join(base_dir, "external", "RAFT-Flow")
    core_path        = os.path.join(raft_flow_path, "flow_core")
    for p in [raft_stereo_path, raft_flow_path]:
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
from modules.flow.flow_map_RAFT import OpticalFlowRAFT
from modules.stereo.stereo_params_YAML import StereoParamsYAML
from modules.stereo.stereo_rectification import StereoRectification
from pipeline.visualization.video_composition import make_stacked_video


# ── Script-level debug knobs ───────────────────────────────────────────────────
single_frame   = False   # True = show one frame interactively; False = batch mode
img_idx        = 960     # frame index used in single-frame mode
render_images  = True
compose_video  = True
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Optical flow visualisation adhoc test")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    flow_ckpt = cfg.flow_checkpoint if os.path.isabs(cfg.flow_checkpoint) \
                else os.path.join(project_root, cfg.flow_checkpoint)

    params        = StereoParamsYAML(cfg.yaml_file)
    rectification = StereoRectification(params)
    flow_solver   = OpticalFlowRAFT(flow_ckpt, rectification, cfg.raft_iters, cfg.raft_optflow_warmstart)

    _, rectification_mask, __, ___ = rectification.get_rectification_masks()

    # ── Single frame mode ──────────────────────────────────────────────────────
    if single_frame:
        frame1 = os.path.join(cfg.dataset_path, f"img/image_0_{img_idx}.png")
        frame2 = os.path.join(cfg.dataset_path, f"img/image_0_{img_idx + 1}.png")

        img1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise ValueError("One or both images not found. Check file paths.")

        flow_uv = flow_solver.compute_flow(img1, img2)

        flow_masked = flow_uv.copy()
        flow_masked[0][~rectification_mask] = 0
        flow_masked[1][~rectification_mask] = 0
        flow_ring = flow_solver.to_image(flow_masked)

        flow_masked[0][~rectification_mask] = np.nan
        flow_masked[1][~rectification_mask] = np.nan

        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        axs[0].imshow(img1, cmap="gray");  axs[0].set_title("Frame 1");        axs[0].axis("off")
        axs[1].imshow(img2, cmap="gray");  axs[1].set_title("Frame 2");        axs[1].axis("off")

        im_u = axs[2].imshow(flow_masked[0])
        axs[2].set_title("Flow U (horizontal)"); axs[2].axis("off")
        fig.colorbar(im_u, ax=axs[2], fraction=0.046, pad=0.04)

        im_v = axs[3].imshow(flow_masked[1])
        axs[3].set_title("Flow V (vertical)"); axs[3].axis("off")
        fig.colorbar(im_v, ax=axs[3], fraction=0.046, pad=0.04)

        axs[4].imshow(flow_ring); axs[4].set_title("Optical Flow"); axs[4].axis("off")

        plt.tight_layout()
        plt.show()
        return

    # ── Batch mode ─────────────────────────────────────────────────────────────
    left_txt = os.path.join(cfg.dataset_path, "left_images.txt")
    df_left  = pd.read_csv(left_txt, sep=r'\s+', comment="#", names=["id", "timestamp", "image_name"])
    left_files = df_left["image_name"].tolist()
    if cfg.limit:
        left_files = left_files[:cfg.limit]

    out_flow_dir = os.path.join(cfg.output_path, "out_flow")

    if render_images:
        print("Rendering optical flow images.")
        os.makedirs(out_flow_dir, exist_ok=True)

        for i in range(len(left_files) - 1):
            img1 = cv2.imread(cfg.dataset_path + left_files[i],     cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(cfg.dataset_path + left_files[i + 1], cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                print(f"Skipping frame {i} (missing file).")
                continue

            flow_uv = flow_solver.compute_flow(img1, img2)

            flow_masked = flow_uv.copy()
            flow_masked[0][~rectification_mask] = 0
            flow_masked[1][~rectification_mask] = 0
            flow_image = flow_solver.to_image(flow_masked)

            index = int(left_files[i].split("_")[-1].split(".")[0])
            plt.imsave(os.path.join(out_flow_dir, f"{index}_flow.png"), flow_image)

            if i % 20 == 0:
                print(f"Processed {i} of {len(left_files) - 1} frames.")

    if compose_video:
        flow_files = left_files[:-1]  # flow[i] = motion from frame i to i+1
        transformations = [
            lambda x: x,
            lambda x: os.path.join("out_flow", f"{int(x.split('_')[-1].split('.')[0])}_flow.png"),
        ]
        make_stacked_video(cfg.dataset_path, flow_files,
                           os.path.join(cfg.output_path, "flow_video.mp4"),
                           transformations, 25, (1, 2))

    print("Processing complete.")


if __name__ == "__main__":
    main()
