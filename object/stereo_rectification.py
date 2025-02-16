import cv2
import numpy as np
from stereo_interfaces import StereoRectificationInterface, StereoParamsInterface


class StereoRectification(StereoRectificationInterface):
    def __init__(self, params: StereoParamsInterface):
        """
        Initializes stereo rectification using camera calibration parameters.

        Args:
            params (StereoParamsInterface): Instance containing stereo calibration data.
        """
        self.params = params
        self.rectification_mask = None
        self.roi_mask = None
        self.R1 = self.R2 = self.P1 = self.P2 = self.Q = None  # ✅ Store matrices
        self._init_rectification_maps()

    def _init_rectification_maps(self):
        """
        Computes the rectification maps and stores rectification matrices.
        """
        K_l, D_l, K_r, D_r, R, T, size = (
            self.params.K_l, self.params.D_l,
            self.params.K_r, self.params.D_r,
            self.params.R, self.params.T,
            self.params.resolution
        )

        # ✅ Compute rectification matrices and store them
        self.R1, self.R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY
        )

        self.map1_l, self.map2_l = cv2.initUndistortRectifyMap(K_l, D_l, self.R1, self.P1, size, cv2.CV_32FC1)
        self.map1_r, self.map2_r = cv2.initUndistortRectifyMap(K_r, D_r, self.R2, self.P2, size, cv2.CV_32FC1)

    def rectify_images(self, img_left, img_right):
        """
        Rectifies the input stereo images and ensures rectification masks are computed.

        Args:
            img_left (numpy.ndarray): Left stereo image.
            img_right (numpy.ndarray): Right stereo image.

        Returns:
            tuple: (rectified_left, rectified_right)
        """
        h, w = img_left.shape[:2]

        # ✅ Lazy initialization of rectification masks
        if self.rectification_mask is None or self.rectification_mask.shape != (h, w):
            print("Computing rectification masks...")
            self.rectification_mask, self.roi_mask = self._compute_masks(h, w)

        # Rectify images
        rectified_left = cv2.remap(img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        return rectified_left, rectified_right

    def get_rectification_masks(self):
        """
        Returns the precomputed rectification masks. If they are not computed yet,
        compute them using the resolution from parameters.

        Returns:
            tuple: (rectification_mask, roi_mask)
        """
        # Lazy computation using resolution from parameters
        if self.rectification_mask is None or self.roi_mask is None:
            print("🔄 Rectification masks not found! Computing now...")
            h, w = self.params.resolution[::-1]  # Ensure correct order (height, width)
            self.rectification_mask, self.roi_mask = self._compute_masks(h, w)

        return self.rectification_mask, self.roi_mask

    def get_rectification_matrices(self):
        """
        Returns the rectification transformation matrices.

        Returns:
            tuple: (R1, R2, P1, P2, Q)
        """
        if None in [self.R1, self.R2, self.P1, self.P2, self.Q]:
            raise RuntimeError("Rectification matrices have not been computed yet.")
        return self.R1, self.R2, self.P1, self.P2, self.Q

    def _compute_masks(self, h, w):
        """
        Computes rectification masks and expands them by 1 pixel in-place.

        Args:
            h (int): Image height.
            w (int): Image width.

        Returns:
            tuple: (rectification_mask, roi_mask)
        """
        dummy_image = np.full((h, w), 255, dtype=np.uint8)

        # Rectify dummy images to detect invalid areas
        rectified_left = cv2.remap(dummy_image, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(dummy_image, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

        # Compute rectification mask (valid pixels)
        self.rectification_mask = (rectified_left > 0) & (rectified_right > 0)

        # Shrink the rectification mask by 1 pixel using morphological erosion
        kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel shrinks by 1 pixel
        self.rectification_mask = cv2.erode(self.rectification_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Compute ROI Mask (region of interest from stereoRectify)
        self.roi_mask = np.zeros_like(self.rectification_mask, dtype=np.uint8)
        roi1, roi2 = cv2.stereoRectify(
            self.params.K_l, self.params.D_l, self.params.K_r, self.params.D_r,
            (w, h), self.params.R, self.params.T, flags=cv2.CALIB_ZERO_DISPARITY
        )[5:7]

        self.roi_mask[roi1[1]:roi1[1] + roi1[3], roi1[0]:roi1[0] + roi1[2]] = 1
        self.roi_mask[roi2[1]:roi2[1] + roi2[3], roi2[0]:roi2[0] + roi2[2]] = 1

        return self.rectification_mask, self.roi_mask

