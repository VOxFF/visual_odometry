import os
import sys

# Get the absolute path of RAFT-Stereo and RAFT-Flow
raft_stereo_path = os.path.join(os.path.dirname(__file__), "external", "RAFT-Stereo")
raft_flow_path = os.path.join(os.path.dirname(__file__), "external", "RAFT-Flow")
sys.path.append(raft_stereo_path)
sys.path.append(raft_flow_path)

core_path = os.path.join(raft_flow_path, "flow_core")
sys.path.insert(0, core_path)  # Ensure core modules are found


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stereo.stereo_interfaces import StereoParamsInterface
from stereo.stereo_depth import StereoDepth
from stereo.stereo_params_YAML import StereoParamsYAML
from stereo.stereo_rectification import StereoRectification
from stereo.stereo_disparity_RAFT import DisparityRAFT
from flow.flow_map_RAFT import OpticalFlowRAFT
from keypoints.keypoints_uniform import UniformKeyPoints
from keypoints.keypoints_3d import Keypoints3DXform
from keypoints.keypoints_3d_flow import Keypoints3DFlow

# Dataset Path (Change to your dataset)
dataset_path = "/home/roman/Downloads/fpv_datasets/indoor_forward_7_snapdragon_with_gt/"
yaml_file = "/home/roman/Downloads/fpv_datasets/indoor_forward_calib_snapdragon/indoor_forward_calib_snapdragon_imu.yaml"

# RAFT Checkpoints
stereo_checkpoint = "/home/roman/Rainbow/visual_odometry/models/raft-stereo/raftstereo-sceneflow.pth"
flow_checkpoint = "/home/roman/Rainbow/visual_odometry/models/rart-flow/raft-things.pth"

# Load calibration parameters
params = StereoParamsYAML(yaml_file)

# Initialize rectification
rectification = StereoRectification(params)

# Initialize RAFT-Stereo for disparity
disparity_solver = DisparityRAFT(stereo_checkpoint, rectification)
depth_solver = StereoDepth(params)

# Initialize RAFT for optical flow
flow_solver = OpticalFlowRAFT(flow_checkpoint, rectification)

# Initialize Keypoint Extraction and 3D Processing
pts_src = UniformKeyPoints(rectification_mask=rectification.get_rectification_masks()[0])
pts_xform = Keypoints3DXform(params.get_camera_params(StereoParamsInterface.StereoCamera.LEFT))
pts_flow = Keypoints3DFlow(params.get_camera_params(StereoParamsInterface.StereoCamera.LEFT), pts_xform, rectification.get_rectification_masks()[0])


# **Single Frame Mode**
img_idx = 600  # Select frame index
frame1 = dataset_path + f"img/image_0_{img_idx}.png"
frame2 = dataset_path + f"img/image_0_{img_idx+1}.png"
right_img = dataset_path + f"img/image_1_{img_idx}.png"

# Load raw images
img1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None or img_right is None:
    raise ValueError("One or more images not found. Check file paths.")

# Compute disparity & depth for frame 1
disparity = disparity_solver.compute_disparity(img1, img_right)
depth1 = depth_solver.compute_depth(disparity)
depth1 = np.clip(depth1, 0, 20)  # Depth clipping

# Compute disparity & depth for frame 2
disparity2 = disparity_solver.compute_disparity(img2, img_right)
depth2 = depth_solver.compute_depth(disparity2)
depth2 = np.clip(depth2, 0, 20)  # Depth clipping

# Compute optical flow
flow_uv = flow_solver.compute_flow(img1, img2)

# Extract keypoints (uniformly distributed)
keypoints = pts_src.get_keypoints(img1, max_number=200)
##
print(keypoints.shape)
min_vals = keypoints.min(axis=0)  # [min_x, min_y]
max_vals = keypoints.max(axis=0)  # [max_x, max_y]

print("min_x:", min_vals[0])
print("min_y:", min_vals[1])
print("max_x:", max_vals[0])
print("max_y:", max_vals[1])

##
# Compute 3D keypoints and motion
keypoints_3d_f1 = pts_xform.to_3d(keypoints, depth1)
keypoints_3d_f2, valid_mask = pts_flow.compute_3d_flow(keypoints, depth1, depth2, flow_uv)

# Project keypoints_3d_f2 back to 2D space
projected_keypoints_f2 = pts_xform.to_2d(keypoints_3d_f2[valid_mask])

# Extract valid keypoints for visualization
keypoints_valid = keypoints[valid_mask]
projected_keypoints_valid = projected_keypoints_f2

# Visualization
fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(img1, cmap="gray")
ax.set_title(f"Optical Flow Vectors - Frame {img_idx} → {img_idx+1}")
ax.axis("off")

# Draw optical flow vectors
for (x1, y1), (x2, y2) in zip(keypoints_valid, projected_keypoints_valid):
    ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=2, head_length=3, color="red")

plt.show()
