import os
import sys

# Get the absolute path of RAFT-Stereo and RAFT-Flow
raft_stereo_path = os.path.join(os.path.dirname(__file__), "external", "RAFT-Stereo")
raft_flow_path = os.path.join(os.path.dirname(__file__), "external", "RAFT-Flow")
sys.path.append(raft_stereo_path)
sys.path.append(raft_flow_path)


stereo_core_path = os.path.join(raft_stereo_path, "core")
sys.path.insert(0, stereo_core_path)  # Ensure core modules are found
from stereo.stereo_disparity_RAFT import DisparityRAFT


# Unload 'core' module from sys.modules to resolve the conflict
if "core" in sys.modules:
    del sys.modules["core"]

if "core.utils" in sys.modules:
    del sys.modules["core.utils"]

sys.path = [p for p in sys.path if "RAFT-Stereo" not in p]
flow_core_path = os.path.join(raft_flow_path, "core")
flow_utils_path = os.path.join(flow_core_path , "utils")
sys.path.insert(0, flow_core_path)  # Ensure core modules are found
sys.path.insert(0, flow_utils_path)

# ✅ Explicitly reload core modules
import core.utils
import core.raft
import core.update
import core.extractor
import core.corr
import utils.utils  # 🚀 Ensure `utils` is loaded correctly

from importlib import reload
reload(core.utils)
reload(core.raft)
reload(core.update)
reload(core.extractor)
reload(core.corr)
reload(utils.utils)  # 🚀 Reload `utils.utils` so it's correctly interpreted

from flow.flow_map_RAFT import OpticalFlowRAFT

for m in sys.modules:
    print(m)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stereo.stereo_depth import StereoDepth
from stereo.stereo_params_YAML import StereoParamsYAML
from stereo.stereo_rectification import StereoRectification

from keypoints.keypoints_uniform import UniformKeypoints
from keypoints.keypoints_3d import Keypoints3D
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
uniform_keypoints = UniformKeypoints(rectification_mask=rectification.get_rectification_masks()[0])
keypoints_3d_transform = Keypoints3D(params.get_camera_parameters(params.StereoCamera.LEFT))
keypoints_flow_solver = Keypoints3DFlow(keypoints_3d_transform, rectification.get_rectification_masks()[0])


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
keypoints = uniform_keypoints.get_keypoints(img1, max_number=200)

# Compute 3D keypoints and motion
keypoints_3d_f1 = keypoints_3d_transform.to_3d(keypoints, depth1)
keypoints_3d_f2, valid_mask = keypoints_flow_solver.compute_3d_flow(keypoints, depth1, depth2, flow_uv)

# Project keypoints_3d_f2 back to 2D space
projected_keypoints_f2 = keypoints_3d_transform.to_2d(keypoints_3d_f2[valid_mask])

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
