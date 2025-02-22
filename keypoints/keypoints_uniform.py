import numpy as np
from keypoints.keypoints_interfaces import KeyPointsInterface

class UniformKeyPoints(KeyPointsInterface):
    """
    Generates uniformly distributed keypoints across an image while discarding
    points that fall outside the rectification mask.
    """

    def __init__(self, rectification_mask: np.ndarray):
        """
        Initializes the keypoint extractor.

        Args:
            rectification_mask (numpy.ndarray): A boolean mask where valid regions are `True`.
        """
        self.rectification_mask = rectification_mask

    def get_keypoints(self, image, max_number: int):
        """
        Generates uniformly distributed keypoints within the valid region of the rectification mask.

        Args:
            image (numpy.ndarray): The input image (ignored in this method).
            max_number (int): The maximum number of keypoints to return.

        Returns:
            numpy.ndarray: An array of shape (N, 2) containing keypoints (x, y) in image-space pixels.
        """
        h, w = self.rectification_mask.shape

        # Generate a uniform grid of keypoints
        num_x = int(np.sqrt(max_number * (w / h)))  # Adjust grid density to aspect ratio
        num_y = int(np.sqrt(max_number * (h / w)))
        xs = np.linspace(0, w - 1, num_x).astype(int)
        ys = np.linspace(0, h - 1, num_y).astype(int)

        # Create a meshgrid and flatten
        grid_x, grid_y = np.meshgrid(xs, ys)
        keypoints = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

        # Filter keypoints using rectification mask
        valid_mask = self.rectification_mask[keypoints[:, 1], keypoints[:, 0]]  # Check mask at (y, x)
        keypoints = keypoints[valid_mask]

        # Limit to max_number keypoints
        if len(keypoints) > max_number:
            keypoints = keypoints[np.linspace(0, len(keypoints) - 1, max_number, dtype=int)]

        return keypoints
