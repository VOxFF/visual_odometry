import cv2
import numpy as np
from stereo_interfaces import StereoRectificationInterface, StereoDisparityInterface

class StereoDisparityOpenCV(StereoDisparityInterface):
    """
    Computes stereo disparity using OpenCV's StereoSGBM or StereoBM.
    Requires a rectification interface for input images.
    """

    def __init__(self, rectification: StereoRectificationInterface, method="SGBM", num_disparities=16, block_size=11):
        """
        Initializes the disparity computation method.

        Args:
            rectification (StereoRectificationInterface): Instance of a rectification implementation.
            method (str): "SGBM" for Semi-Global Matching, "BM" for Block Matching.
            num_disparities (int): Number of disparity levels (must be divisible by 16).
            block_size (int): Block size for matching.
        """
        if not isinstance(rectification, StereoRectificationInterface):
            raise ValueError("rectification must be an instance of StereoRectificationInterface")

        if num_disparities % 16 != 0:
            raise ValueError("num_disparities must be a multiple of 16.")

        self.rectification = rectification
        self.method = method
        self.num_disparities = num_disparities
        self.block_size = block_size

        if method == "BM":
            self.stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        elif method == "SGBM":
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8 * 3 * block_size**2,
                P2=32 * 3 * block_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=5,
                speckleWindowSize=50,
                speckleRange=2,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        else:
            raise ValueError("Invalid method. Choose 'SGBM' or 'BM'.")

    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """
        Computes the disparity map after rectifying input images.

        Args:
            img_left (numpy.ndarray): Left input image.
            img_right (numpy.ndarray): Right input image.

        Returns:
            numpy.ndarray: Normalized disparity map.
        """
        if img_left is None or img_right is None:
            raise ValueError("One or both images were not found or invalid.")

        # Rectify images using the rectification interface
        rectified_left, rectified_right = self.rectification.rectify_images(img_left, img_right)

        # Compute disparity
        disparity = self.stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
        return disparity
