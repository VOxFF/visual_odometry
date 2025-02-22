import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Tuple

class KeyPointsInterface(ABC):
    """
    Abstract interface for keypoint detection.
    """

    @abstractmethod
    def get_keypoints(self, image: np.ndarray, max_number: int) -> np.ndarray:
        """
        Extracts keypoints from the input image.

        Args:
            image (np.ndarray): Grayscale image.
            max_number (int): Maximum number of keypoints to detect.

        Returns:
            np.ndarray: Array of detected keypoints (N, 2).
        """
        pass


class Keypoints3DInterface(ABC):
    """
    Interface for computing 3D keypoints from depth and projecting them back to the image plane.
    """

    @abstractmethod
    def to_3d(self, keypoints: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Converts 2D keypoints to 3D world coordinates using the depth map.

        Args:
            keypoints (np.ndarray): Array of shape (N, 2) with (u, v) pixel coordinates.
            depth_map (np.ndarray): Depth map with the same resolution as the image.

        Returns:
            np.ndarray: Array of shape (N, 3) with (X, Y, Z) 3D coordinates in camera space.
        """
        pass

    @abstractmethod
    def to_2d(self, points_3D: np.ndarray) -> np.ndarray:
        """
        Projects 3D points back to the 2D image plane.

        Args:
            points_3D (np.ndarray): Array of shape (N, 3) with (X, Y, Z) 3D coordinates in camera space.

        Returns:
            np.ndarray: Array of shape (N, 2) with (u, v) pixel coordinates.
        """
        pass



class Keypoints3DFlowInterface(ABC):
    """ Interface for computing 2D and 3D keypoint flow. """

    @abstractmethod
    def compute_2d_flow(self, keypoints: np.ndarray, uv_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the new 2D locations of keypoints after applying optical flow.

        Args:
            keypoints (np.ndarray): Array of shape (N, 2) with (u, v) keypoints.
            uv_flow (np.ndarray): Optical flow map of shape (H, W, 2).

        Returns:
            tuple:
                - np.ndarray: Updated keypoints of shape (N, 2).
                - np.ndarray: Validity mask of shape (N,), True for valid points.
        """
        pass

    @abstractmethod
    def compute_3d_flow(
        self, keypoints: np.ndarray, depth1: np.ndarray, depth2: np.ndarray, uv_flow: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes new 3D keypoints at frame (f+1) after applying optical flow.

        Args:
            keypoints (np.ndarray): Array of shape (N, 2) with (u, v) keypoints.
            depth1 (np.ndarray): Depth map at frame f (H, W).
            depth2 (np.ndarray): Depth map at frame f+1 (H, W).
            uv_flow (np.ndarray): Optical flow map of shape (H, W, 2).

        Returns:
            tuple:
                - np.ndarray: New 3D keypoints at frame f+1 of shape (N, 3).
                - np.ndarray: Validity mask of shape (N,), True for valid points.
        """
        pass
