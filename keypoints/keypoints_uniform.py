import numpy as np
from keypoints.keypoints_interfaces import KeyPointsInterface

class UniformKeyPoints(KeyPointsInterface):
    """
    Generates uniformly distributed keypoints across an image while discarding
    points that fall outside the rectification mask. An optional margin can be
    specified to restrict the keypoints to an inner region of the image.
    """

    def __init__(self, rectification_mask: np.ndarray, margin: int = 0):
        """
        Initializes the keypoint extractor.

        Args:
            rectification_mask (numpy.ndarray): A boolean mask where valid regions are True.
            margin (int): Margin in pixels to exclude from the image border (default: 0).
        """
        self.rectification_mask = rectification_mask
        self.margin = margin

    def get_keypoints(self, image, max_number: int):
        """
        Generates uniformly distributed keypoints within the valid region of the rectification mask,
        restricted by the specified margin.

        Args:
            image (numpy.ndarray): The input image (ignored in this method).
            max_number (int): The maximum number of keypoints to return.

        Returns:
            numpy.ndarray: An array of shape (N, 2) containing keypoints (x, y) in image-space pixels.
        """
        h, w = self.rectification_mask.shape

        # Define the valid region based on the margin.
        x0, x1 = self.margin, w - self.margin
        y0, y1 = self.margin, h - self.margin

        # Generate a uniform grid of keypoints within the valid region.
        # Adjust grid density based on the aspect ratio of the valid region.
        num_x = int(np.sqrt(max_number * ((x1 - x0) / (y1 - y0))))
        num_y = int(np.sqrt(max_number * ((y1 - y0) / (x1 - x0))))
        xs = np.linspace(x0, x1 - 1, num_x).astype(int)
        ys = np.linspace(y0, y1 - 1, num_y).astype(int)

        # Create a meshgrid and flatten.
        grid_x, grid_y = np.meshgrid(xs, ys)
        keypoints = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

        # Filter keypoints using the rectification mask.
        valid_mask = self.rectification_mask[keypoints[:, 1], keypoints[:, 0]]
        keypoints = keypoints[valid_mask]

        # Limit to max_number keypoints.
        if len(keypoints) > max_number:
            keypoints = keypoints[np.linspace(0, len(keypoints) - 1, max_number, dtype=int)]

        return keypoints
