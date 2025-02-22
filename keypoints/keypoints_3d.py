import numpy as np
from stereo.stereo_interfaces import CameraParametersInterface
from keypoints.keypoints_interfaces import Keypoints3DInterface

class Keypoints3DXform(Keypoints3DInterface):
    """
    Implementation of Keypoints3DInterface for converting 2D keypoints to 3D and projecting them back.
    """

    def __init__(self, camera_params: CameraParametersInterface):
        """
        Initializes Keypoints3D with camera parameters.

        Args:
            camera_params (CameraParametersInterface): The camera intrinsic parameters.
        """
        self.camera_params = camera_params
        self.K_inv = np.linalg.inv(self.camera_params.K)  # Precompute inverse intrinsic matrix

    def to_3d(self, keypoints: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Converts 2D keypoints to 3D coordinates using the provided 2D depth map and camera intrinsics.

        For each keypoint (u, v), the function retrieves the corresponding depth value Z from the
        depth_map using the indices [int(v), int(u)]. If Z is positive, the keypoint is transformed
        into 3D coordinates by applying the inverse of the intrinsic camera matrix and scaling by Z.
        If Z is non-positive, the function assigns a default value (e.g., [0, 0, 0]) for that keypoint.

        Args:
            keypoints (np.ndarray): An array of shape (N, 2) containing 2D keypoints (u, v) in image coordinates.
            depth_map (np.ndarray): A 2D array (H, W) of depth values corresponding to the image grid.

        Returns:
            np.ndarray: An array of shape (N, 3) containing the 3D coordinates (X, Y, Z) for each valid keypoint.
        """
        points_3D = []
        # Ensure depth_map is 2D
        if depth_map.ndim != 2:
            raise ValueError("depth_map must be a 2D array.")
        for (u, v) in keypoints:
            Z = depth_map[int(v), int(u)]
            if Z <= 0:
                points_3D.append([0, 0, 0])
            else:
                uv_homogeneous = np.array([u, v, 1.0])
                xy = self.K_inv @ uv_homogeneous
                X, Y = xy[:2] * Z
                points_3D.append([X, Y, Z])
        return np.array(points_3D)

    def to_2d(self, points_3D: np.ndarray) -> np.ndarray:
        """
        Projects 3D points back to the 2D image plane.

        Args:
            points_3D (np.ndarray): Array of shape (N, 3) with (X, Y, Z) 3D coordinates in camera space.

        Returns:
            np.ndarray: Array of shape (N, 2) with (u, v) pixel coordinates.
        """
        keypoints_2D = []
        for X, Y, Z in points_3D:
            if Z <= 0:  # Avoid division by zero
                continue

            uv_homogeneous = self.camera_params.K @ np.array([X / Z, Y / Z, 1.0])
            u, v = uv_homogeneous[:2]
            keypoints_2D.append([u, v])

        return np.array(keypoints_2D)
