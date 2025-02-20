import numpy as np
from stereo.stereo_interfaces import CameraParametersInterface
from keypoints.keypoints_interfaces import Keypoints3DInterface

class Keypoints3D(Keypoints3DInterface):
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

    def keypoints_to_3D(self, keypoints: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Converts 2D keypoints to 3D world coordinates using the depth map.

        Args:
            keypoints (np.ndarray): Array of shape (N, 2) with (u, v) pixel coordinates.
            depth_map (np.ndarray): Depth map with the same resolution as the image.

        Returns:
            np.ndarray: Array of shape (N, 3) with (X, Y, Z) 3D coordinates in camera space.
        """
        points_3D = []
        for (u, v) in keypoints:
            Z = depth_map[int(v), int(u)]  # Get depth value at (u, v)
            if Z <= 0:  # Skip invalid depth values
                continue

            uv_homogeneous = np.array([u, v, 1.0])  # Convert to homogeneous coordinates
            xy = self.K_inv @ uv_homogeneous  # Apply inverse intrinsics
            X, Y = xy[:2] * Z  # Scale by depth

            points_3D.append([X, Y, Z])

        return np.array(points_3D)

    def project_3D_to_image(self, points_3D: np.ndarray) -> np.ndarray:
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
