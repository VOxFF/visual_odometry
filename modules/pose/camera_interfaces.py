from abc import ABC, abstractmethod
import numpy as np


class CameraXformInterface(ABC):
    """
    Interface for computing the camera transformation (rigid transformation) between two sets of 3D points.
    This transformation includes only rotation and translation.
    """

    @abstractmethod
    def compute_camera_xform(self, P: np.ndarray, Q: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Compute the camera transformation that aligns a set of 3D points P to the corresponding set Q.

        Args:
            P (np.ndarray): An array of shape (N, 3) containing 3D points from the current frame.
            Q (np.ndarray): An array of shape (N, 3) containing corresponding 3D points from the next frame.

        Returns:
            tuple:
                - R (np.ndarray): A 3x3 rotation matrix.
                - t (np.ndarray): A translation vector of length 3.
        """
        pass