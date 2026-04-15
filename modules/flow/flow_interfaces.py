from abc import ABC, abstractmethod
import numpy as np

class OpticalFlowInterface(ABC):
    """
    Abstract interface for optical flow computation.
    """

    @abstractmethod
    def compute_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Computes the optical flow between two images.

        Args:
            img1 (numpy.ndarray): First input image (grayscale or RGB).
            img2 (numpy.ndarray): Second input image.

        Returns:
            numpy.ndarray: Optical flow map (shape: H x W x 2) with (dx, dy) motion vectors.
        """
        pass
