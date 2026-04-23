import numpy as np
from scipy.ndimage import map_coordinates
from modules.stereo.stereo_interfaces import CameraParametersInterface
from modules.keypoints.keypoints_interfaces import Keypoints3DInterface


def _bilinear_sample(map2d: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a 2D map at float (u, v) coordinates."""
    return map_coordinates(map2d, [vs, us], order=1, mode='nearest')


class Keypoints3DXform(Keypoints3DInterface):
    """
    Implementation of Keypoints3DInterface for converting 2D keypoints to 3D and projecting them back.
    """

    def __init__(self, camera_params: CameraParametersInterface, subpixel: bool = False):
        """
        Initializes Keypoints3D with camera parameters.

        Args:
            camera_params (CameraParametersInterface): The camera intrinsic parameters.
            subpixel (bool): Use bilinear depth sampling at float coordinates instead of nearest-integer.
        """
        self.camera_params = camera_params
        self.K_inv = np.linalg.inv(self.camera_params.K)
        self.subpixel = subpixel

    def to_3d(self, keypoints: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Converts 2D keypoints to 3D coordinates using the provided 2D depth map and camera intrinsics.

        Args:
            keypoints (np.ndarray): An array of shape (N, 2) containing 2D keypoints (u, v) in image coordinates.
            depth_map (np.ndarray): A 2D array (H, W) of depth values corresponding to the image grid.

        Returns:
            np.ndarray: An array of shape (N, 3) containing the 3D coordinates (X, Y, Z) for each valid keypoint.
        """
        if depth_map.ndim != 2:
            raise ValueError("depth_map must be a 2D array.")

        if self.subpixel:
            Z = _bilinear_sample(depth_map, keypoints[:, 0], keypoints[:, 1])
        else:
            us = keypoints[:, 0].astype(int)
            vs = keypoints[:, 1].astype(int)
            Z = depth_map[vs, us]

        # Back-project all points at once: K_inv @ [u, v, 1]^T, then scale by depth
        uv_h = np.column_stack([keypoints, np.ones(len(keypoints))])  # (N, 3)
        xyz = (self.K_inv @ uv_h.T).T                                  # (N, 3)
        points_3D = xyz * Z[:, None]

        # Zero out points with invalid (non-positive) depth
        points_3D[Z <= 0] = 0.0

        return points_3D

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
