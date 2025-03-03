import os
import sys
import cv2
import re
import ast
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Add RAFT-Stereo and RAFT-Flow paths
raft_stereo_path = os.path.join(os.path.dirname(__file__), "external", "RAFT-Stereo")
raft_flow_path = os.path.join(os.path.dirname(__file__), "external", "RAFT-Flow")
sys.path.append(raft_stereo_path)
sys.path.append(raft_flow_path)
core_path = os.path.join(raft_flow_path, "flow_core")
sys.path.insert(0, core_path)

# Import necessary modules from your project.
from stereo.stereo_interfaces import StereoParamsInterface
from stereo.stereo_depth import StereoDepth
from stereo.stereo_params_YAML import StereoParamsYAML
from stereo.stereo_rectification import StereoRectification
from stereo.stereo_disparity_RAFT import DisparityRAFT
from flow.flow_map_RAFT import OpticalFlowRAFT
from keypoints.keypoints_uniform import UniformKeyPoints
from keypoints.keypoints_3d import Keypoints3DXform
from keypoints.keypoints_3d_flow import Keypoints3DFlow
from camera.camera_svd_xform import CameraSvdXform
from camera.camera_svd_xform import CameraRansacXform
from utilities.video_composition import make_stacked_video
from utilities.data_utils import match_ground_truth_positions
from utilities.data_utils import read_ground_truth_positions
from utilities.data_utils import read_ground_truth_transforms
from utilities.plot_3d import TrajectoryPlot

# ------------------------------
# Configuration and Initialization
# ------------------------------

dataset_path = "/home/roman/Downloads/fpv_datasets/indoor_forward_7_snapdragon_with_gt/"
yaml_file = "/home/roman/Downloads/fpv_datasets/indoor_forward_calib_snapdragon/indoor_forward_calib_snapdragon_imu.yaml"

# RAFT checkpoints
stereo_checkpoint = "/home/roman/Rainbow/visual_odometry/models/raft-stereo/raftstereo-sceneflow.pth"
flow_checkpoint = "/home/roman/Rainbow/visual_odometry/models/rart-flow/raft-things.pth"

# To do
compute_trajectory = True
render_images = True
compose_movie = True

# Parameters.
limit = 0  # Use 0 for no limit.

min_depth = 0.0  # meters
max_depth = 15.0  # meters


# Load calibration parameters and initialize rectification.
params = StereoParamsYAML(yaml_file)
rectification = StereoRectification(params)

# Initialize solvers.
disparity_solver = DisparityRAFT(stereo_checkpoint, rectification)
depth_solver = StereoDepth(params)
flow_solver = OpticalFlowRAFT(flow_checkpoint, rectification)

# Initialize keypoint extraction and 3D processing.
# Retrieve the rectification masks (assuming they have been computed already)
stereo_mask, left_mask, right_mask, roi_mask = rectification.get_rectification_masks()
pts_src = UniformKeyPoints(stereo_mask)
pts_xform = Keypoints3DXform(params.get_camera_params(StereoParamsInterface.StereoCamera.LEFT))
pts_flow = Keypoints3DFlow(params.get_camera_params(StereoParamsInterface.StereoCamera.LEFT),
                           pts_xform, stereo_mask)


# Initialize the camera transformation estimator.
#cam_estimator = CameraSvdXform(offset=np.array([0.02172, -6.61e-05, -0.00049]))
#cam_estimator = CameraSvdXform()
cam_estimator = CameraRansacXform()

# Prepare file and folder paths.
traj_txt_path = os.path.join(dataset_path, "camera_trajectory.txt")
truth_txt_path = os.path.join(dataset_path, "groundtruth.txt")
traj_img_dir = os.path.join(dataset_path, "out_traj")
os.makedirs(traj_img_dir, exist_ok=True)
output_dir = os.path.join(dataset_path, "out_cam_tracking")
os.makedirs(output_dir, exist_ok=True)

# Read left image filenames.
left_txt = os.path.join(dataset_path, "left_images.txt")
df_left = pd.read_csv(left_txt, delim_whitespace=True, comment="#",
                      names=["id", "timestamp", "image_name"])
left_files = df_left["image_name"].tolist()
if limit:
    left_files = left_files[:limit]

# ------------------------------
# Compute Trajectory and Write to Text File (Optimized with Cache Check)
# ------------------------------
if compute_trajectory:
    traj_file = open(traj_txt_path, "w")
    traj_file.write("frame, translation, rotation_matrix_flat\n")

    T_global = np.eye(4)  # Global camera pose (4x4 homogeneous); start at identity.


    # Cache variables for the "previous" frame results.
    prev_img_left = None
    prev_img_right = None
    prev_depth = None

    print("Computing trajectory over frames...")
    for i in range(len(left_files) - 1):
        # For the first iteration, prev_depth will be None.
        if prev_depth is None:
            frame1_path = os.path.join(dataset_path, left_files[i])
            frame2_path = os.path.join(dataset_path, left_files[i + 1])
            img_left1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
            img_left2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
            if img_left1 is None or img_left2 is None:
                print(f"Skipping frame {i} due to missing images.")
                continue

            right1_path = frame1_path.replace("image_0_", "image_1_")
            right2_path = frame2_path.replace("image_0_", "image_1_")
            img_right1 = cv2.imread(right1_path, cv2.IMREAD_GRAYSCALE)
            img_right2 = cv2.imread(right2_path, cv2.IMREAD_GRAYSCALE)
            if img_right1 is None or img_right2 is None:
                print(f"Skipping frame {i} due to missing right images.")
                continue

            # Compute disparity and depth for both frames.
            disp1 = disparity_solver.compute_disparity(img_left1, img_right1)
            depth1 = depth_solver.compute_depth(disp1)
            disp2 = disparity_solver.compute_disparity(img_left2, img_right2)
            depth2 = depth_solver.compute_depth(disp2)
            # Compute optical flow from img1 to img2.
            flow_uv = flow_solver.compute_flow(img_left1, img_left2)

            # Cache the second frame's data for the next iteration.
            prev_img_left = img_left2
            prev_img_right = img_right2
            prev_depth = depth2
        else:
            # For subsequent iterations, use cached previous frame.
            img_left1 = prev_img_left
            depth1 = prev_depth

            # Load new current frame (frame i+1).
            frame2_path = os.path.join(dataset_path, left_files[i + 1])
            img_left2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
            if img_left2 is None:
                print(f"Skipping frame {i} due to missing image for frame {i+1}.")
                continue
            right2_path = frame2_path.replace("image_0_", "image_1_")
            img_right2 = cv2.imread(right2_path, cv2.IMREAD_GRAYSCALE)
            if img_right2 is None:
                print(f"Skipping frame {i} due to missing right image for frame {i+1}.")
                continue

            # Compute disparity and depth for the new current frame.
            disp2 = disparity_solver.compute_disparity(img_left2, img_right2)
            depth2 = depth_solver.compute_depth(disp2)
            # Compute optical flow from the cached frame to the new frame.
            flow_uv = flow_solver.compute_flow(img_left1, img_left2)

            # Update the cache.
            prev_img_left = img_left2
            prev_img_right = img_right2
            prev_depth = depth2

        # Keypoint reinitialization/tracking.
        keypoints_2D = pts_src.get_keypoints(img_left1, max_number=320)
        keypoints_3D_1 = pts_xform.to_3d(keypoints_2D, depth1)

        # Let's try to limit the range
        valid_depth_mask = (keypoints_3D_1[:, 2] >= min_depth) & (keypoints_3D_1[:, 2] <= max_depth)
        keypoints_3D_1 = keypoints_3D_1[valid_depth_mask]
        keypoints_2D = keypoints_2D[valid_depth_mask]
        print(f"keypoints = {len(keypoints_2D)}")

        keypoints_3D_2, valid_mask = pts_flow.compute_3d_flow(keypoints_2D, depth1, depth2, flow_uv)

        # Check if we have enough valid keypoints.
        if keypoints_3D_2.shape[0] < 4:
            print("Too few valid keypoints; skipping transformation update.")
            continue

        # Filter the keypoints using the valid mask.
        old_3D = keypoints_3D_1[valid_mask]
        new_3D = keypoints_3D_2[valid_mask]

        # Create a mask where dz <= 1 add filter
        dz = new_3D[:, 2] - old_3D[:, 2]
        dz_mask = abs(dz) <= 1.0
        old_3D = old_3D[dz_mask]
        new_3D = new_3D[dz_mask]


        # Compute the relative transformation (rotation and translation) that maps old_3D to new_3D.
        R_rel, t_rel = cam_estimator.compute_camera_xform(old_3D, new_3D)
        T_rel = np.eye(4)
        T_rel[:3, :3] = R_rel
        T_rel[:3, 3] = t_rel

        # Update the global camera pose.
        T_global = T_global @ T_rel

        # Write the current frame's transformation to file.
        R_flat = R_rel.flatten()
        traj_file.write(f"{i}, {t_rel.tolist()}, {R_flat.tolist()}\n")

        if i % 20 == 0:
            print(f"Processed {i} / {len(left_files) - 1} frames.")

    traj_file.close()
    print("Trajectory computation complete. Data written to:", traj_txt_path)


# ------------------------------
# Render Trajectory Images with Fixed Scale
# ------------------------------
if render_images:

    # Customization parameters.
    show_axis = False  # If False, tick labels will be hidden.
    small_font_size = 8  # Use a smaller font size.
    zoom_distance = 5    # Lower value zooms in (default is typically around 10).

    # elevation = 0        # For left view
    # azimuth = 90

    elevation = 90         # For top view
    azimuth = 0

    # elevation = 45  # Regular view
    # azimuth = -45


    # Initialize the global transformation as identity (no rotation, no translation)
    T_global = np.eye(4)
    global_positions = []   # to store camera positions
    global_Ts = []          # to store full 4x4 transformation matrices
    # matrix for z/z swap  [x, y, z] into [x, z, –y].
    M = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]])

    print("Rendering images")
    with open(traj_txt_path, 'r') as f:
        header = f.readline()  # Skip the header line.
        lines = itertools.islice(f, limit) if limit and limit > 0 else f
        for line in lines:
            # Use regex to extract both bracketed lists from the line.
            matches = re.findall(r'\[.*?\]', line)
            if len(matches) >= 2:
                # The first list is the translation vector.
                translation = ast.literal_eval(matches[0])
                # The second list is the flattened rotation matrix.
                rotation_flat = ast.literal_eval(matches[1])
                # Reshape the flattened list into a 3x3 matrix.
                R_rel = np.array(rotation_flat).reshape(3, 3)
                t_rel = np.array(translation)

                # Create the relative transformation matrix (4x4 homogeneous).
                T_rel = np.eye(4)
                T_rel[:3, :3] = R_rel
                T_rel[:3, 3] = t_rel

                # Update the global transformation: T_global = T_global * T_rel.
                T_global = T_global @ T_rel

                cam_pos = T_global[:3, 3].copy()  # original camera position
                R_cam = T_global[:3, :3]  # original rotation matrix

                world_pos = M @ cam_pos  # convert translation
                R_world = M @ R_cam @ M.T  # convert rotation (note: for a permutation matrix, M.T == M⁻¹)

                T_world = np.eye(4)
                T_world[:3, :3] = R_world
                T_world[:3, 3] = world_pos

                global_positions.append(world_pos)
                global_Ts.append(T_world)


    # Get the combined trajectory (list of tuples) by matching ground truth.
    #combined_positions = match_ground_truth_positions(global_positions, left_txt, truth_txt_path)
    if False:
        global_positions = read_ground_truth_positions(truth_txt_path, skip=20)
        global_Ts = read_ground_truth_transforms(truth_txt_path, skip=20)

    if len(global_positions) > 0:
        print("Rendering trajectory images with fixed scale...")

        # Create an instance of your TrajectoryPlot class.
        from utilities.plot_3d import TrajectoryPlot

        tp = TrajectoryPlot(global_positions, elevation=elevation, azimuth=azimuth,
                            zoom_distance=zoom_distance, small_font_size=small_font_size)

        # For each frame, render the trajectory up to that frame using its full transformation.
        for idx in range(len(global_positions)):
            current_T = global_Ts[idx]  # Use the transformation for the current frame.
            fig = tp.plot(current_T, idx)
            plot_path = os.path.join(traj_img_dir, f"traj_{idx:06d}.png")
            fig.savefig(plot_path)
            plt.close(fig)

            if idx % 20 == 0:
                print(f"Rendered trajectory image for frame {idx + 1} / {len(global_positions)}")
    else:
        print("No global positions available for trajectory rendering.")


# ------------------------------
# Compose Movie: Original Left Image + Trajectory Visualization
# ------------------------------
if compose_movie:
    print("Composing movie from tracking outputs.")
    # Define transformation lambdas:
    # First column: original left image (using dataset_path and the file name from left_files)
    # Second column: trajectory plot from traj_img_dir.
    transformations = [
        lambda x: os.path.join(dataset_path, x),  # Original left image path.
        lambda x: os.path.join(traj_img_dir, f"traj_{int(x.split('_')[-1].split('.')[0]):06d}.png"),
    ]
    make_stacked_video(dataset_path, left_files, "cam_tracking_video.mp4", transformations)
    print(f"Movie composed as {dataset_path}/cam_tracking_video.mp4")

print("Processing complete.")
