import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from stereo_paramsYAML import StereoParamsYAML
from stereo_rectification import StereoRectification
from stereo_disparity_openCV import StereoDisparityOpenCV
from stereo_depth import StereoDepth

"""
https://fpv.ifi.uzh.ch/datasets/
"""

# Outdoor dataset
# dataset_path = "/home/roman/Downloads/fpv_datasets/outdoor_forward_1_snapdragon_with_gt/"
# yaml_file = "/home/roman/Downloads/fpv_datasets/outdoor_forward_calib_snapdragon/camchain-imucam-outdoor_forward_calib_snapdragon_imu.yaml"
# left_file = dataset_path + "img/image_0_2148.png"
# right_file = dataset_path + "img/image_1_2148.png"

# Indoor dataset
dataset_path = "/home/roman/Downloads/fpv_datasets/indoor_forward_7_snapdragon_with_gt/"
yaml_file = "/home/roman/Downloads/fpv_datasets/indoor_forward_calib_snapdragon/indoor_forward_calib_snapdragon_imu.yaml"
# left_file = dataset_path + "img/image_0_748.png"
# right_file = dataset_path + "img/image_1_748.png"

# left_file = dataset_path + "img/image_0_1325.png"
# right_file = dataset_path + "img/image_1_1325.png"

left_file = dataset_path + "img/image_0_2923.png"
right_file = dataset_path + "img/image_1_2923.png"

# Single image mode
use_single_image = True
limit = 100  # Set to None for full dataset


# Load stereo calibration parameters
params = StereoParamsYAML(yaml_file)

# Initialize rectification using calibration parameters
rectification = StereoRectification(params)

# Initialize disparity and depth computation (moved outside of loop)
disparity_computer = StereoDisparityOpenCV(rectification, method="SGBM", num_disparities=128, block_size=5)
depth_computer = StereoDepth(params)

# Ensure output directories exist
out_disp_path = dataset_path + "out_disp/"
out_depth_path = dataset_path + "out_depth/"
os.makedirs(out_disp_path, exist_ok=True)
os.makedirs(out_depth_path, exist_ok=True)

# Function to normalize disparity/depth maps for visualization
def normalize(img, min_val, max_val):
    return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Process single image pair
if use_single_image:
    img_left = cv2.imread(left_file, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_file, cv2.IMREAD_GRAYSCALE)

    # img_left = cv2.Laplacian(img_left, cv2.CV_8U, ksize=3)
    # img_right = cv2.Laplacian(img_right, cv2.CV_8U, ksize=3)

    if img_left is None or img_right is None:
        raise ValueError("One or both images were not found. Check your file paths.")

    disparity = disparity_computer.compute_disparity(img_left, img_right)
    depth = depth_computer.compute_depth(disparity)

    # Display results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].imshow(img_left, cmap="gray")
    axs[0, 0].set_title("Original Left Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(img_right, cmap="gray")
    axs[0, 1].set_title("Original Right Image")
    axs[0, 1].axis("off")

    im_disp = axs[1, 0].imshow(disparity, cmap="plasma")
    axs[1, 0].set_title("Disparity Map")
    axs[1, 0].axis("off")
    fig.colorbar(im_disp, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im_depth = axs[1, 1].imshow(depth, cmap="inferno")
    axs[1, 1].set_title("Depth Map")
    axs[1, 1].axis("off")
    fig.colorbar(im_depth, ax=axs[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# Process sequence of images
else:

    left_txt, right_txt = dataset_path + "left_images.txt", dataset_path + "right_images.txt"

    # Read the file, ignoring comments (#) in the header
    df_left = pd.read_csv(left_txt, delim_whitespace=True, comment="#", names=["id", "timestamp", "image_name"])
    df_right = pd.read_csv(right_txt, delim_whitespace=True, comment="#", names=["id", "timestamp", "image_name"])

    left_files = df_left["image_name"].tolist()
    right_files = df_right["image_name"].tolist()

    if limit:
        left_files = left_files[:limit]
        right_files = right_files[:limit]

    for i, (left_img, right_img) in enumerate(zip(left_files, right_files)):
        left_path = dataset_path + left_img
        right_path = dataset_path + right_img

        img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        if img_left is None or img_right is None:
            print(f"Skipping missing image pair: {left_img}, {right_img}")
            continue

        disparity = disparity_computer.compute_disparity(img_left, img_right)
        depth = depth_computer.compute_depth(disparity)

        # Extract index from filename
        index = int(left_img.split("_")[-1].split(".")[0])

        # Save normalized disparity and depth
        disparity_normalized = normalize(disparity, np.percentile(disparity, 2), np.percentile(disparity, 98))
        depth_normalized = normalize(depth, np.percentile(depth, 2), np.percentile(depth, 98))

        plt.imsave(out_disp_path + f"{index}_disparity.png", disparity_normalized, cmap="jet")
        plt.imsave(out_depth_path + f"{index}_depth.png", depth_normalized, cmap="inferno")

        if i % 20 == 0:
            print(f"Processed {i} of {len(left_files)} images.")

print("Processing complete.")
