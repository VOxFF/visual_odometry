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
from utilities.video_composition import make_stacked_video

# Dataset Path (Change to your dataset)
dataset_path = "/home/roman/Downloads/fpv_datasets/indoor_forward_7_snapdragon_with_gt/"
yaml_file = "/home/roman/Downloads/fpv_datasets/indoor_forward_calib_snapdragon/indoor_forward_calib_snapdragon_imu.yaml"

# RAFT Checkpoints
stereo_checkpoint = "/home/roman/Rainbow/visual_odometry/models/raft-stereo/raftstereo-sceneflow.pth"
flow_checkpoint = "/home/roman/Rainbow/visual_odometry/models/rart-flow/raft-things.pth"

single_frame = True

# Single frame params
#img_idx = 50
#img_idx = 300
#img_idx = 400
#img_idx = 401
#img_idx = 600
img_idx = 960
# img_idx = 1200
# img_idx = 2000
# img_idx = 2800

ghosting = False
z_labels = False

# Multi frame params
render_images = False
compose_movie = True
limit = 0
k = 25

# Tracks which start is in range are red, otherwise blue
min_dist = 0.1  # meters
max_dist = 6.0  # meters -- it looks like it's noisy after this

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

print(f"Maximal detectable z={params.get_z_max()}")

# **Single Frame Mode**
if single_frame:
    f1_left = dataset_path + f"img/image_0_{img_idx}.png"
    f2_left = dataset_path + f"img/image_0_{img_idx + 1}.png"
    f1_right = dataset_path + f"img/image_1_{img_idx}.png"
    f2_right = dataset_path + f"img/image_1_{img_idx + 1}.png"

    # Load raw images
    img1_left = cv2.imread(f1_left, cv2.IMREAD_GRAYSCALE)
    img2_left = cv2.imread(f2_left, cv2.IMREAD_GRAYSCALE)
    img1_right = cv2.imread(f1_right, cv2.IMREAD_GRAYSCALE)
    img2_right = cv2.imread(f2_right, cv2.IMREAD_GRAYSCALE)

    if img1_left is None or img2_left is None or img1_right is None:
        raise ValueError("One or more images not found. Check file paths.")

    # Compute disparity & depth for frame 1
    disparity = disparity_solver.compute_disparity(img1_left, img1_right)
    depth1 = depth_solver.compute_depth(disparity)
    #depth1 = np.clip(depth1, 0, 20)  # Depth clipping

    # Compute disparity & depth for frame 2
    disparity2 = disparity_solver.compute_disparity(img2_left, img2_right)
    depth2 = depth_solver.compute_depth(disparity2)
    #depth2 = np.clip(depth2, 0, 20)  # Depth clipping

    # Compute optical flow
    flow_uv = flow_solver.compute_flow(img1_left, img2_left)

    # Extract keypoints (uniformly distributed)
    keypoints = pts_src.get_keypoints(img1_left, max_number=200)
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

    # Project keypoints_3d_f2 back to 2D space.
    projected_keypoints_f2 = pts_xform.to_2d(keypoints_3d_f2[valid_mask])

    # Extract valid 2D keypoints for visualization.
    keypoints_valid = keypoints[valid_mask]
    projected_keypoints_valid = projected_keypoints_f2

    # Also extract the valid 3D keypoints from frame 1.
    keypoints_3d_valid_f1 = keypoints_3d_f1[valid_mask]
    keypoints_3d_valid_f2 = keypoints_3d_f2[valid_mask]


    # Visualization
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img1_left, cmap="gray")
    #ax.imshow(depth2, cmap="inferno")
    #ax.imshow(-disparity, cmap="inferno")
    #ax.imshow(img2_right, cmap="gray")


    # Optionally overlay the next frame for ghosting.
    if ghosting:
        ax.imshow(img2_left, cmap="gray", alpha=0.3)

    ##
    dz_n =keypoints_3d_valid_f1[:, 2], keypoints_3d_valid_f2[:, 2]
    print(f"dz max: {np.max(dz_n)} min: {np.min(dz_n)}")
    ##

    # Draw flow vectors with arrow color based on depth range.
    for i in range(len(keypoints_valid)):
        x1, y1 = keypoints_valid[i]
        x2, y2 = projected_keypoints_valid[i]


        # Retrieve depth (z) from frame 1's valid 3D keypoints.
        depth_value = keypoints_3d_valid_f1[i, 2]
        z1, z2 = keypoints_3d_valid_f1[i, 2], keypoints_3d_valid_f2[i, 2]
        dz = z2 - z1

        if abs(dz) > 1:
            print(f"{i} dz max: {dz}")

        # Determine arrow color: red if depth is within [min_dist, max_dist], else blue.
        color = "yellow" if min_dist <= depth_value <= max_dist else "blue"
        ax.arrow(x1, y1, (x2 - x1), (y2 - y1), head_width=2, head_length=3, color=color)

        if z_labels:
            color = "yellow" if abs(dz) < 1 else "red"
            ax.text(x1 + 4, y1 + 4, f"{dz:.2f}", color=color, fontsize=8)
            #ax.text(x1 + 4, y1 + 4, f"{z1:.2f}:{z2:.2f}", color=color, fontsize=8)

    ax.set_title(f"Optical Flow Vectors - Frame {img_idx} → {img_idx + 1}")
    ax.axis("off")
    plt.show()


else:
    # **Multi Frame Mode**
    print("Multi frame")
    output_dir = "out_flow_points"

    left_txt = os.path.join(dataset_path, "left_images.txt")
    df_left = pd.read_csv(left_txt, delim_whitespace=True, comment="#", names=["id", "timestamp", "image_name"])
    left_files = df_left["image_name"].tolist()
    if limit:
        left_files = left_files[:limit]

    out_folder = os.path.join(dataset_path, output_dir)
    os.makedirs(out_folder, exist_ok=True)

    # This will store the 2D tracks with depth information.
    tracks_2D = []  # Each track is a list of (x, y, z) tuples.
    track_start_z = []  # Store the starting depth for each track.

    # Also store the current 2D keypoints and current 3D keypoints.
    current_2D_keypoints = None
    current_3D_points = None

    if render_images:
        for i in range(len(left_files) - 1):
            # Load the current and next images.
            frame1_path = os.path.join(dataset_path, left_files[i])
            frame2_path = os.path.join(dataset_path, left_files[i + 1])
            img1_left = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
            img2_left = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
            if img1_left is None or img2_left is None:
                continue

            # Also load the corresponding right images for depth computation.
            right1_path = frame1_path.replace("image_0_", "image_1_")
            right2_path = frame2_path.replace("image_0_", "image_1_")
            img_right1 = cv2.imread(right1_path, cv2.IMREAD_GRAYSCALE)
            img_right2 = cv2.imread(right2_path, cv2.IMREAD_GRAYSCALE)
            if img_right1 is None or img_right2 is None:
                continue

            # Compute disparity/depth for frame i.
            disp1 = disparity_solver.compute_disparity(img1_left, img_right1)
            depth1 = depth_solver.compute_depth(disp1)

            # Compute disparity/depth for frame i+1.
            disp2 = disparity_solver.compute_disparity(img2_left, img_right2)
            depth2 = depth_solver.compute_depth(disp2)

            # If this is the start of a new k-segment, reinitialize keypoints & tracks.
            if i % k == 0 or current_2D_keypoints is None:
                # Detect 2D keypoints in the current frame.
                new_keypoints_2D = pts_src.get_keypoints(img1_left, max_number=200)
                # Convert them to 3D.
                current_3D_points = pts_xform.to_3d(new_keypoints_2D, depth1)
                valid_3D_mask = (current_3D_points[:, 2] > 0)
                current_3D_points = current_3D_points[valid_3D_mask]
                current_2D_keypoints = new_keypoints_2D[valid_3D_mask]

                # Initialize the track list and starting depths.
                tracks_2D = []
                track_start_z = []
                for pt2d, pt3d in zip(current_2D_keypoints, current_3D_points):
                    tracks_2D.append([(pt2d[0], pt2d[1], pt3d[2])])
                    track_start_z.append(pt3d[2])
            else:
                # Compute optical flow from frame i to i+1.
                flow_uv = flow_solver.compute_flow(img1_left, img2_left)
                new_3D_points, valid_mask = pts_flow.compute_3d_flow(current_2D_keypoints, depth1, depth2, flow_uv)
                if new_3D_points.shape[0] < 4:
                    new_keypoints_2D = pts_src.get_keypoints(img1_left, max_number=200)
                    current_3D_points = pts_xform.to_3d(new_keypoints_2D, depth1)
                    valid_3D_mask = (current_3D_points[:, 2] > 0)
                    current_3D_points = current_3D_points[valid_3D_mask]
                    current_2D_keypoints = new_keypoints_2D[valid_3D_mask]
                    tracks_2D = []
                    track_start_z = []
                    for pt2d, pt3d in zip(current_2D_keypoints, current_3D_points):
                        tracks_2D.append([(pt2d[0], pt2d[1], pt3d[2])])
                        track_start_z.append(pt3d[2])
                    continue

                new_3D_points = new_3D_points[valid_mask]
                new_2D_keypoints = pts_xform.to_2d(new_3D_points)
                current_3D_points = new_3D_points
                current_2D_keypoints = new_2D_keypoints

                # Update tracks: for each valid keypoint, append the new (x, y, z) point.
                new_tracks = []
                new_track_start_z = []
                idx_valid = 0
                for j, was_valid in enumerate(valid_mask):
                    if was_valid:
                        new_pt = (current_2D_keypoints[idx_valid][0], current_2D_keypoints[idx_valid][1],
                                  current_3D_points[idx_valid, 2])
                        new_tracks.append(tracks_2D[j] + [new_pt])
                        new_track_start_z.append(track_start_z[j])
                        idx_valid += 1
                tracks_2D = new_tracks
                track_start_z = new_track_start_z

            # Visualization: Draw each track with segment color based on depth change.
            # Define depth range limits.
            min_dist = 2.0  # meters
            max_dist = 10.0  # meters


            # Helper: determine segment color based on starting depth and current depth.
            def get_color(start_depth, current_depth):
                depth_change = current_depth - start_depth
                if min_dist <= depth_change <= max_dist:
                    return (0, 0, 255)  # Red (BGR).
                else:
                    return (255, 0, 0)  # Blue (BGR).


            vis_img = cv2.cvtColor(img1_left, cv2.COLOR_GRAY2BGR)
            for track, start_depth in zip(tracks_2D, track_start_z):
                if len(track) < 2:
                    continue
                for seg_idx in range(len(track) - 1):
                    pt1 = track[seg_idx]
                    pt2 = track[seg_idx + 1]
                    color = get_color(start_depth, pt2[2])
                    cv2.line(vis_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness=2)

            # Write the visualization image to file.
            out_filename = f"{i:06d}.png"
            out_path = os.path.join(out_folder, out_filename)
            cv2.imwrite(out_path, vis_img)

            if i % 20 == 0:
                print(f"Processed {i} / {len(left_files) - 1} frames")

    if compose_movie:
        print("Composing movie.")
        n = len(left_files)
        left_files = left_files[:n - 1]

        # Define transformation lambdas
        transformations = [
            lambda x: f"{output_dir}/{int(x.split('_')[-1].split('.')[0]):06d}.png",
            #lambda x: f"{output_dir}/{int(x.split('_')[-1].split('.')[0])}_flow.png",
        ]

        # Generate stacked video
        make_stacked_video(dataset_path, left_files, "keypoints_video.mp4", transformations)


print("Processing complete.")

