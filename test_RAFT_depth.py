

import os
import sys

# Get the absolute path of RAFT-Stereo
raft_path = os.path.join(os.path.dirname(__file__), "external", "RAFT-Stereo")
sys.path.append(raft_path)  # Add RAFT to Python path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stereo.stereo_disparity_RAFT import DisparityRAFT
from stereo.stereo_depth import StereoDepth
from stereo.stereo_params_YAML import StereoParamsYAML
from stereo.stereo_rectification import StereoRectification
from utilities.video_composition import make_stacked_video


# Outdoor dataset
#dataset_path = "/home/roman/Downloads/fpv_datasets/outdoor_forward_1_snapdragon_with_gt/"
#yaml_file = "/home/roman/Downloads/fpv_datasets/outdoor_forward_calib_snapdragon/camchain-imucam-outdoor_forward_calib_snapdragon_imu.yaml"

# Indoor dataset
dataset_path = "/home/roman/Downloads/fpv_datasets/indoor_forward_7_snapdragon_with_gt/"
yaml_file = "/home/roman/Downloads/fpv_datasets/indoor_forward_calib_snapdragon/indoor_forward_calib_snapdragon_imu.yaml"

checkpoint = "/home/roman/Rainbow/visual_odometry/models/raft-stereo/raftstereo-sceneflow.pth"

# Depth clipping limits - not sure that we need this
DEPTH_MIN = 0
DEPTH_MAX = 20

# Load calibration parameters
params = StereoParamsYAML(yaml_file)

# Initialize rectification
rectification = StereoRectification(params)

# Pass rectification to DisparityRAFT
disparity_solver = DisparityRAFT(checkpoint, rectification)

# Initialize depth solver (StereoDepth)
depth_solver = StereoDepth(params)

# Load mask
mask = cv2.imread("/home/roman/Downloads/fpv_datasets/mask.png", cv2.IMREAD_GRAYSCALE).astype(np.uint8) == 0


single_frame = True  # Set to True for testing a single frame

# Multi-frame options
render_images = False
compose_video = True
limit = 0  # Set to None for full dataset

# **Single Frame Mode**
if single_frame:
    # img_idx = 2800
    #img_idx = 2100
    #img_idx = 1200
    #img_idx = 600
    img_idx = 400
    left_file = dataset_path + f"img/image_0_{img_idx}.png"
    right_file = dataset_path + f"img/image_1_{img_idx}.png"

    # Load raw images
    img_left = cv2.imread(left_file, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_file, cv2.IMREAD_GRAYSCALE)

    # Compute disparity & depth
    disparity = disparity_solver.compute_disparity(img_left, img_right)
    depth = depth_solver.compute_depth(disparity)
    rectification_mask, _, __, ___ = rectification.get_rectification_masks()

    # Apply depth clipping
    depth = np.clip(depth, DEPTH_MIN, DEPTH_MAX)

    # Apply rectification mask
    disparity_masked = disparity.copy()
    depth_masked = depth.copy()
    disparity_masked[~rectification_mask] = np.nan  # Use NaN for visualization
    depth_masked[~rectification_mask] = np.nan  # Use NaN for visualization

    # Display images
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].imshow(img_left, cmap="gray")
    axs[0, 0].set_title("Left Image")
    axs[0, 1].imshow(img_right, cmap="gray")
    axs[0, 1].set_title("Right Image")

    im_disp = axs[1, 0].imshow(np.where(mask, -disparity_masked, np.nan), cmap="jet")
    axs[1, 0].set_title("Disparity Map (Masked)")
    axs[1, 0].axis("off")
    fig.colorbar(im_disp, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im_depth = axs[1, 1].imshow(np.where(mask, depth_masked, np.nan), cmap="inferno")
    axs[1, 1].set_title("Metric Depth Map (Clipped & Masked)")
    axs[1, 1].axis("off")
    fig.colorbar(im_depth, ax=axs[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# **Batch Processing Mode**
else:
    left_txt, right_txt = dataset_path + "left_images.txt", dataset_path + "right_images.txt"

    # Read filenames, ignoring comments (#)
    df_left = pd.read_csv(left_txt, delim_whitespace=True, comment="#", names=["id", "timestamp", "image_name"])
    df_right = pd.read_csv(right_txt, delim_whitespace=True, comment="#", names=["id", "timestamp", "image_name"])

    left_files = df_left["image_name"].tolist()
    right_files = df_right["image_name"].tolist()
    if limit:
        left_files = left_files[:limit]
        right_files = right_files[:limit]

    # Render frames
    if render_images:
        print("Rendering images.")

        os.makedirs(dataset_path + "out_disp/", exist_ok=True)
        os.makedirs(dataset_path + "out_depth/", exist_ok=True)

        for i, (left_img, right_img) in enumerate(zip(left_files, right_files)):
            left_path = dataset_path + left_img
            right_path = dataset_path + right_img

            # Load raw images
            img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
            img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

            disparity = disparity_solver.compute_disparity(img_left, img_right)
            depth = depth_solver.compute_depth(disparity)
            rectification_mask, _, __, ___ = rectification.get_rectification_masks()

            # Apply depth clipping
            depth = np.clip(depth, DEPTH_MIN, DEPTH_MAX)

            # Apply rectification mask for saving
            disparity_saved = disparity.copy()
            depth_saved = depth.copy()
            disparity_saved[~rectification_mask] = 0  # Use 0 for saving
            depth_saved[~rectification_mask] = 0  # Use 0 for saving

            index = int(left_img.split("_")[-1].split(".")[0])  # Extract frame index

            # Save disparity & depth
            plt.imsave(dataset_path + f"out_disp/{index}_disparity.png", np.where(mask, -disparity_saved, 0), cmap="jet")
            plt.imsave(dataset_path + f"out_depth/{index}_depth.png", np.where(mask, depth_saved, 0), cmap="inferno")

            if i % 20 == 0:
                print(f"Processed {i} of {len(left_files)} images.")

    # Compose video from images
    if compose_video:

        # Define transformation lambdas
        transformations = [
            lambda x: x,  # Original image
            lambda x: x.replace("_0.png", "_1.png"),  # Right image
            lambda x: f"out_disp/{int(x.split('_')[-1].split('.')[0])}_disparity.png", # Disparity image
            lambda x: f"out_depth/{int(x.split('_')[-1].split('.')[0])}_depth.png"    # Depht image
        ]

        # Generate stacked video
        make_stacked_video(dataset_path, left_files, "depth_video.mp4", transformations)

print("Processing complete.")
