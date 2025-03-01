import numpy as np
from stereo.stereo_interfaces import StereoParamsInterface

class StereoDepth:
    """
    Computes depth from disparity using camera intrinsics and baseline.
    """

    def __init__(self, params: StereoParamsInterface):
        """
        Initializes the depth estimation using stereo calibration parameters.

        Args:
            params (StereoParamsInterface): An implementation of stereo calibration parameters.
        """
        if not isinstance(params, StereoParamsInterface):
            raise ValueError("params must be an instance of StereoParamsInterface")

        self.fx = params.focal_length_px
        self.baseline = params.get_baseline()

        if self.fx <= 0 or self.baseline <= 0:
            raise ValueError("Focal length and baseline must be positive values.")

    def compute_depth(self, disparity):
        """
        Computes depth from disparity.

        Args:
            disparity (numpy.ndarray): Disparity map.

        Returns:
            tuple: (depth_map, valid_mask)
                - depth_map (numpy.ndarray): Computed depth values.
                - valid_mask (numpy.ndarray): Binary mask where depth is valid.
        """
        if disparity is None:
            raise ValueError("Disparity map is None.")

        # Create a mask where disparity is valid (greater than zero)
        valid = np.abs(disparity) > 0

        # Initialize depth map
        depth = np.zeros_like(disparity, dtype=np.float32)

        # Compute depth only where disparity is valid
        depth[valid] = (self.fx * self.baseline) / abs(disparity[valid])

        return depth
