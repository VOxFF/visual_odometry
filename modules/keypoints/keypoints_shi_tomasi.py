import cv2
import numpy as np
from modules.keypoints.keypoints_interfaces import KeyPointsInterface


class ShiTomasiKeyPoints(KeyPointsInterface):
    """
    Detects keypoints using Shi-Tomasi corner detection (cv2.goodFeaturesToTrack).
    Points outside the rectification mask are discarded.
    """

    def __init__(self, rectification_mask: np.ndarray, quality_level: float = 0.01, min_distance: int = 7):
        """
        Args:
            rectification_mask (np.ndarray): Boolean mask (H, W), True for valid rectified regions.
            quality_level (float): Minimum accepted quality of corners (0–1).
            min_distance (int): Minimum pixel distance between returned corners.
        """
        self.rectification_mask = rectification_mask
        self.quality_level = quality_level
        self.min_distance = min_distance

    def get_keypoints(self, image: np.ndarray, max_number: int) -> np.ndarray:
        """
        Detects Shi-Tomasi corners within the valid rectification region.

        Args:
            image (np.ndarray): Grayscale input image.
            max_number (int): Maximum number of keypoints to return.

        Returns:
            np.ndarray: Array of shape (N, 2) with (u, v) keypoint coordinates.
        """
        mask = self.rectification_mask.astype(np.uint8) * 255

        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=max_number,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )

        if corners is None:
            return np.empty((0, 2), dtype=np.float32)

        return corners.reshape(-1, 2)
