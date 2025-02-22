import numpy as np
from typing import Tuple
from keypoints.keypoints_interfaces import Keypoints3DInterface
from keypoints.keypoints_interfaces  import Keypoints3DFlowInterface
from stereo.stereo_interfaces import CameraParametersInterface


class Keypoints3DFlow(Keypoints3DFlowInterface):
    def __init__(self, camera_params: CameraParametersInterface, keypoints_xform: Keypoints3DInterface,
                 rectification_mask: np.ndarray):
        """
        Initializes Keypoints3DFlow.

        Args:
            camera_params (CameraParametersInterface): Camera intrinsic parameters.
            keypoints_xform (Keypoints3DInterface): Keypoints 3D transformation instance.
            rectification_mask (np.ndarray): Boolean mask (H, W), True for valid rectified regions.
        """
        self.camera_params = camera_params
        self.keypoints_xform = keypoints_xform  # ✅ Uses Keypoints3DInterface for 3D transformations
        self.rectification_mask = rectification_mask

    def compute_2d_flow(self, keypoints: np.ndarray, uv_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the new 2D positions of keypoints after applying optical flow.

        Args:
            keypoints (np.ndarray): Nx2 array of (u, v) pixel coordinates.
            uv_flow (np.ndarray): Optical flow field (H, W, 2) with (du, dv) vectors.

        Returns:
            tuple:
                - np.ndarray: Updated keypoints of shape (N, 2).
                - np.ndarray: Validity mask of shape (N,), True for valid points.
        """
        uv_flow = np.transpose(uv_flow, (1, 2, 0)) #transpose to W,H,2
        keypoints_next = keypoints + uv_flow[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)]

        # Ensure keypoints remain inside the image bounds
        h, w = uv_flow.shape[:2]
        valid_mask = (keypoints_next[:, 0] >= 0) & (keypoints_next[:, 0] < w) & \
                     (keypoints_next[:, 1] >= 0) & (keypoints_next[:, 1] < h)

        keypoints_next[:, 0] = np.clip(keypoints_next[:, 0], 0, w - 1)
        keypoints_next[:, 1] = np.clip(keypoints_next[:, 1], 0, h - 1)

        return keypoints_next, valid_mask

    def compute_3d_flow(self, keypoints: np.ndarray, depth1: np.ndarray, depth2: np.ndarray, uv_flow: np.ndarray) -> \
    Tuple[np.ndarray, np.ndarray]:
        """
        Computes the 3D motion of keypoints using optical flow and depth maps.

        Args:
            keypoints (np.ndarray): Nx2 array of (u, v) pixel coordinates.
            depth1 (np.ndarray): Depth map for frame f (H, W).
            depth2 (np.ndarray): Depth map for frame f+1 (H, W).
            uv_flow (np.ndarray): Optical flow field (H, W, 2) with (du, dv) vectors.

        Returns:
            tuple:
                - np.ndarray: Updated 3D keypoints of shape (N, 3).
                - np.ndarray: Validity mask of shape (N,), True for valid points.
        """
        # Compute new 2D positions using optical flow
        keypoints_next, valid_mask = self.compute_2d_flow(keypoints, uv_flow)

        # Extract per-keypoint depth values (for validity checking only)
        depth_vals1 = depth1[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)]
        depth_vals2 = depth2[keypoints_next[:, 1].astype(int), keypoints_next[:, 0].astype(int)]

        # Create a valid mask ensuring that both depth values are positive
        depth_valid_mask = valid_mask & (depth_vals1 > 0) & (depth_vals2 > 0)

        # Initialize output arrays for 3D keypoints
        keypoints_3d = np.zeros((keypoints.shape[0], 3))
        keypoints_3d_next = np.zeros((keypoints.shape[0], 3))

        if np.any(depth_valid_mask):
            # IMPORTANT: Pass the full 2D depth maps to to_3d
            keypoints_3d[depth_valid_mask] = self.keypoints_xform.to_3d(
                keypoints[depth_valid_mask], depth1
            )
            keypoints_3d_next[depth_valid_mask] = self.keypoints_xform.to_3d(
                keypoints_next[depth_valid_mask], depth2
            )

        return keypoints_3d_next, depth_valid_mask


