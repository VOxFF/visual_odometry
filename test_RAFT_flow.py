import os
import sys

# Get the absolute path of RAFT (Optical Flow)
raft_path = os.path.join(os.path.dirname(__file__), "external", "RAFT-Flow")
sys.path.append(raft_path)  # Add RAFT to Python path

core_path = os.path.join(raft_path, "flow_core")
sys.path.insert(0, core_path)  # Ensure core modules are found

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flow.flow_map_RAFT import OpticalFlowRAFT
from stereo.stereo_params_YAML import StereoParamsYAML
from stereo.stereo_rectification import StereoRectification
from utilities.video_composition import make_stacked_video


# Outdoor dataset
# dataset_path = "/home/roman/Downloads/fpv_datasets/outdoor_forward_1_snapdragon_with_gt/"
# yaml_file = "/home/roman/Downloads/fpv_datasets/outdoor_forward_calib_snapdragon/camchain-imucam-outdoor_forward_calib_snapdragon_imu.yaml"

# Indoor dataset
dataset_path = "/home/roman/Downloads/fpv_datasets/indoor_forward_7_snapdragon_with_gt/"
yaml_file = "/home/roman/Downloads/fpv_datasets/indoor_forward_calib_snapdragon/indoor_forward_calib_snapdragon_imu.yaml"

# RAFT Optical Flow checkpoint
checkpoint = "/home/roman/Rainbow/visual_odometry/models/rart-flow/raft-things.pth"      #good results but noisy for still frames
#checkpoint = "/home/roman/Rainbow/visual_odometry/models/rart-flow/raft-kitti.pth"      #less no motin noise but distorted on normal frames
#checkpoint = "/home/roman/Rainbow/visual_odometry/models/rart-flow/raft-small.pth"      #needs paramter tweaking
#checkpoint = "/home/roman/Rainbow/visual_odometry/models/rart-flow/raft-sintel.pth"     #still noisy for still frames
#checkpoint = "/home/roman/Rainbow/visual_odometry/models/rart-flow/raft-chairs.pth"     #less noisy for still but confused with shadows

single_frame = True

# Multi-frame options
render_images = True
compose_video = True
limit = 0  # Set to None for full dataset

# Load calibration parameters
params = StereoParamsYAML(yaml_file)

# Initialize rectification
rectification = StereoRectification(params)

# Load Optical Flow Solver (RAFT)
flow_solver = OpticalFlowRAFT(checkpoint, rectification)

if single_frame:
    # Select image pair
    #img_idx = 0
    #img_idx = 50
    #img_idx = 600
    img_idx = 1200
    #img_idx = 2000
    #img_idx = 2800
    frame1 = dataset_path + f"img/image_0_{img_idx}.png"
    frame2 = dataset_path + f"img/image_0_{img_idx+1}.png"  # Next frame in sequence

    # Load raw images
    img1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("One or both images not found. Check file paths.")

    # Compute optical flow
    flow_uv = flow_solver.compute_flow(img1, img2)
    _, rectification_mask, __, ___ = rectification.get_rectification_masks()

    # Apply rectification mask
    flow_uv_masked = flow_uv.copy()
    flow_uv_masked[0][~rectification_mask] = 0 # Use 0 for compuation
    flow_uv_masked[1][~rectification_mask] = 0 # Use 0 for computation

    flow_ring = flow_solver.to_image(flow_uv_masked)

    flow_uv_masked[0][~rectification_mask] = np.nan  # Use NaN for visualization
    flow_uv_masked[1][~rectification_mask] = np.nan  # Use NaN for visualization

    # Visualization
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))

    axs[0].imshow(img1, cmap="gray")
    axs[0].set_title("Frame 1")
    axs[0].axis("off")

    axs[1].imshow(img2, cmap="gray")
    axs[1].set_title("Frame 2")
    axs[1].axis("off")

    # Optical Flow U (Horizontal)
    im_flow_u = axs[2].imshow(flow_uv_masked[0], cmap=None)
    axs[2].set_title("Optical Flow U")
    axs[2].axis("off")
    fig.colorbar(im_flow_u, ax=axs[2], fraction=0.046, pad=0.04)

    # Optical Flow V (Vertical)
    im_flow_v = axs[3].imshow(flow_uv_masked[1], cmap=None)
    axs[3].set_title("Optical Flow V")
    axs[3].axis("off")
    fig.colorbar(im_flow_v, ax=axs[3], fraction=0.046, pad=0.04)

    # Full Optical Flow
    im_flow = axs[4].imshow(flow_ring, cmap=None)
    axs[4].set_title("Optical Flow")
    axs[4].axis("off")
    fig.colorbar(im_flow, ax=axs[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

else:
    left_txt = dataset_path + "left_images.txt"

    # Read filenames, ignoring comments (#)
    df_left = pd.read_csv(left_txt, delim_whitespace=True, comment="#", names=["id", "timestamp", "image_name"])
    left_files = df_left["image_name"].tolist()

    if limit:
        left_files = left_files[:limit]

    rectification_mask = None

    if render_images:
        print("Rendering images.")

        os.makedirs(dataset_path + "out_flow/", exist_ok=True)

        for i in range(len(left_files) - 1):
            frame1 = dataset_path + left_files[i]
            frame2 = dataset_path + left_files[i + 1]

            # Load raw images
            img1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                print(f"Skipping {frame1} and {frame2} (missing file)")
                continue

            # Compute optical flow
            flow_uv = flow_solver.compute_flow(img1, img2)

            if rectification_mask is None:
                _, rectification_mask, __, ___ = rectification.get_rectification_masks()

            # Apply rectification mask
            flow_uv_masked = flow_uv.copy()
            flow_uv_masked[0][~rectification_mask] = 0
            flow_uv_masked[1][~rectification_mask] = 0

            # Convert flow to image representation
            flow_image = flow_solver.to_image(flow_uv_masked)

            # Extract frame index
            index = int(left_files[i].split("_")[-1].split(".")[0])

            # Save flow visualization
            plt.imsave(dataset_path + f"out_flow/{index}_flow.png", flow_image, cmap=None)

            if i % 20 == 0:
                print(f"Processed {i} of {len(left_files)} frames.")

    if compose_video:
        n = len(left_files)
        left_files = left_files[:n-1]

        # Define transformation lambdas
        transformations = [
            lambda x: x,  # Original image
            lambda x: f"out_flow/{int(x.split('_')[-1].split('.')[0])}_flow.png",
        ]

        # Generate stacked video
        make_stacked_video(dataset_path, left_files, "flow_video.mp4", transformations)


print("Processing complete.")