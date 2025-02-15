import cv2
import numpy as np
from stereo_interfaces import StereoRectificationInterface, StereoParamsInterface


class StereoRectification(StereoRectificationInterface):
    """
    Implements StereoRectificationInterface for stereo image rectification.
    Uses an external implementation of StereoParamsInterface for calibration parameters.
    """

    def __init__(self, params: StereoParamsInterface):
        """
        Initializes rectification with stereo calibration parameters.

        Args:
            params (StereoParamsInterface): An implementation of stereo calibration parameters.
        """
        if not isinstance(params, StereoParamsInterface):
            raise ValueError("params must be an instance of StereoParamsInterface")

        self.params = params  # Store calibration parameters implementation

        # Compute rectification matrices
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.params.K_l, self.params.D_l, self.params.K_r, self.params.D_r,
            self.params.resolution, self.params.R, self.params.T
        )

        # Compute rectification maps for fast remapping
        self.map_lx, self.map_ly = cv2.initUndistortRectifyMap(
            self.params.K_l, self.params.D_l, self.R1, self.P1, self.params.resolution, cv2.CV_32FC1
        )
        self.map_rx, self.map_ry = cv2.initUndistortRectifyMap(
            self.params.K_r, self.params.D_r, self.R2, self.P2, self.params.resolution, cv2.CV_32FC1
        )

    def rectify_images(self, img_left, img_right):
        """
        Rectifies stereo images using stored calibration parameters.

        Args:
            img_left (numpy.ndarray): Left image.
            img_right (numpy.ndarray): Right image.

        Returns:
            tuple: (rectified_left, rectified_right)
        """
        rectified_left = cv2.remap(img_left, self.map_lx, self.map_ly, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.map_rx, self.map_ry, cv2.INTER_LINEAR)
        return rectified_left, rectified_right

    def get_rectification_matrices(self):
        """
        Returns rectification matrices.

        Returns:
            dict: Dictionary containing rectification matrices.
        """
        return {
            "R1": self.R1,
            "R2": self.R2,
            "P1": self.P1,
            "P2": self.P2,
            "Q": self.Q
        }
