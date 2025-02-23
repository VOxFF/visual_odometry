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

single_frame = False

# Single frame params
img_idx = 50
#img_idx = 600
#img_idx = 1200
#img_idx = 2000
#img_idx = 2800

ghosting = False

# Multi frame params
limit = 0
k = 25

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
if single_frame:
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
    #depth1 = np.clip(depth1, 0, 20)  # Depth clipping

    # Compute disparity & depth for frame 2
    disparity2 = disparity_solver.compute_disparity(img2, img_right)
    depth2 = depth_solver.compute_depth(disparity2)
    #depth2 = np.clip(depth2, 0, 20)  # Depth clipping

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

    # Overlay Frame f+1 with partial opacity (30%)
    if ghosting:
        ax.imshow(img2, cmap="gray", alpha=0.3)

    # Draw flow vectors
    for (x1, y1), (x2, y2) in zip(keypoints_valid, projected_keypoints_valid):
        ax.arrow(
            x1, y1,
            (x2 - x1), (y2 - y1),
            head_width=2, head_length=3, color="red"
        )

    ax.set_title(f"Optical Flow Vectors - Frame {img_idx} → {img_idx+1}")
    ax.axis("off")
    plt.show()

else:
    # **Muti Frame Frame Mode**

    left_txt = os.path.join(dataset_path, "left_images.txt")
    df_left = pd.read_csv(left_txt, delim_whitespace=True, comment="#", names=["id", "timestamp", "image_name"])
    left_files = df_left["image_name"].tolist()
    if limit:
        left_files = left_files[:limit]

    out_folder = os.path.join(dataset_path, "out_movie_3dflow")
    os.makedirs(out_folder, exist_ok=True)

    # This will store the 2D tracks for visualization, but behind the scenes we will do 3D updates.
    tracks_2D = []  # each entry is a list of (x, y) for each keypoint

    # Also store the "current 2D keypoints" that match each 3D point
    current_2D_keypoints = None
    current_3D_points = None

    for i in range(len(left_files) - 1):
        # Load the current and next images
        frame1_path = os.path.join(dataset_path, left_files[i])
        frame2_path = os.path.join(dataset_path, left_files[i + 1])
        img1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue

        # Also load the corresponding right images so we can compute disparity/depth for both frames
        # (Assuming you have "image_1_xxx.png" as well)
        right1_path = frame1_path.replace("image_0_", "image_1_")
        right2_path = frame2_path.replace("image_0_", "image_1_")
        img_right1 = cv2.imread(right1_path, cv2.IMREAD_GRAYSCALE)
        img_right2 = cv2.imread(right2_path, cv2.IMREAD_GRAYSCALE)
        if img_right1 is None or img_right2 is None:
            continue

        # Compute disparity/depth for frame i
        disp1 = disparity_solver.compute_disparity(img1, img_right1)
        depth1 = depth_solver.compute_depth(disp1)

        # Compute disparity/depth for frame i+1
        disp2 = disparity_solver.compute_disparity(img2, img_right2)
        depth2 = depth_solver.compute_depth(disp2)

        # If this is the start of a new k-segment, re-initialize the keypoints & tracks
        if i % k == 0 or current_2D_keypoints is None:
            # Detect 2D keypoints in the current frame
            new_keypoints_2D = pts_src.get_keypoints(img1, max_number=200)

            # Convert them to 3D
            current_3D_points = pts_xform.to_3d(new_keypoints_2D, depth1)

            # Some 3D might be invalid (Z=0 or negative). Keep only valid ones
            valid_3D_mask = (current_3D_points[:, 2] > 0)
            current_3D_points = current_3D_points[valid_3D_mask]
            current_2D_keypoints = new_keypoints_2D[valid_3D_mask]

            # Initialize the 2D track list: each track is a list of 2D points
            tracks_2D = []
            for pt in current_2D_keypoints:
                tracks_2D.append([tuple(pt)])  # start each track with the current 2D point

        # Compute optical flow from frame i to i+1
        flow_uv = flow_solver.compute_flow(img1, img2)

        # Now update the 3D points from frame i to frame i+1 using compute_3d_flow
        # But note: compute_3d_flow requires 2D keypoints + depth1, depth2, uv_flow
        # We'll do it in a loop or vectorized form:

        # new_3D_points, valid_mask = pts_flow.compute_3d_flow(current_2D_keypoints, depth1, depth2, flow_uv)
        # Instead of "keypoints", we use the "current_2D_keypoints" from last iteration

        new_3D_points, valid_mask = pts_flow.compute_3d_flow(
            current_2D_keypoints, depth1, depth2, flow_uv
        )

        # Keep only the valid ones
        new_3D_points = new_3D_points[valid_mask]
        old_2D_keypoints = current_2D_keypoints[valid_mask]

        # Reproject the new 3D points to get their 2D locations in frame i+1
        new_2D_keypoints = pts_xform.to_2d(new_3D_points)

        # Also store them for the next iteration
        current_3D_points = new_3D_points
        current_2D_keypoints = new_2D_keypoints

        # We must also update the track lists accordingly, dropping those that are invalid
        # Easiest is to build a new list of track arrays:
        new_tracks_2D = []
        idx_valid = 0
        for track, was_valid in zip(tracks_2D, valid_mask):
            if was_valid:
                # append the new point to this track
                track.append(tuple(new_2D_keypoints[idx_valid]))
                new_tracks_2D.append(track)
                idx_valid += 1
            else:
                # this track is no longer valid
                pass
        tracks_2D = new_tracks_2D

        # Visualize
        vis_img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

        for track in tracks_2D:
            pts = np.array(track, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

        # Save the frame
        out_filename = f"{i:06d}.png"
        out_path = os.path.join(out_folder, out_filename)
        cv2.imwrite(out_path, vis_img)

        if i % 20 == 0:
            print(f"Processed {i} / {len(left_files) - 1} frames")

print("Processing complete.")

