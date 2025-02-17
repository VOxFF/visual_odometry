import os
import sys

# Get the absolute path of RAFT (Optical Flow)
raft_path = os.path.join(os.path.dirname(__file__), "external", "RAFT-Flow")
sys.path.append(raft_path)  # Add RAFT to Python path

core_path = os.path.join(raft_path, "core")
sys.path.insert(0, core_path)  # Ensure core modules are found


import cv2
import numpy as np
import matplotlib.pyplot as plt

from flow.flow_map_RAFT import OpticalFlowRAFT


# Dataset Path (Change to your dataset)
dataset_path = "/home/roman/Downloads/fpv_datasets/indoor_forward_7_snapdragon_with_gt/"

# RAFT Optical Flow checkpoint
checkpoint = "/home/roman/Rainbow/visual_odometry/models/rart-flow/raft-things.pth"

# Load Optical Flow Solver (RAFT)
flow_solver = OpticalFlowRAFT(checkpoint)

# Single Frame Mode
single_frame = True  # Set to True for testing a single frame

if single_frame:
    # Select image pair
    img_idx = 600  # Change as needed
    frame1 = dataset_path + f"img/image_0_{img_idx}.png"
    frame2 = dataset_path + f"img/image_0_{img_idx+1}.png"  # Next frame in sequence

    # Load raw images
    img1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("One or both images not found. Check file paths.")

    # Compute optical flow
    flow_map = flow_solver.compute_flow(img1, img2)

    # Extract flow components (dx, dy)
    flow_dx = flow_map[..., 0]
    flow_dy = flow_map[..., 1]
    flow_magnitude = np.sqrt(flow_dx**2 + flow_dy**2)  # Compute magnitude

    # Visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img1, cmap="gray")
    axs[0].set_title("Frame 1")
    axs[0].axis("off")

    axs[1].imshow(img2, cmap="gray")
    axs[1].set_title("Frame 2")
    axs[1].axis("off")

    im_flow = axs[2].imshow(flow_map[1], cmap="jet")
    axs[2].set_title("Optical Flow Magnitude")
    axs[2].axis("off")
    fig.colorbar(im_flow, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
