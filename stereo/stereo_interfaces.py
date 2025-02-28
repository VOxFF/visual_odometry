from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

class CameraParametersInterface(ABC):
    """ Abstract interface for camera parameters. """

    @abstractmethod
    def get_intrinsics(self) -> np.ndarray:
        """ Returns the intrinsic camera matrix. """
        pass

    @abstractmethod
    def get_distortion_coeffs(self) -> np.ndarray:
        """ Returns the distortion coefficients. """
        pass

    @abstractmethod
    def get_resolution(self) -> tuple:
        """ Returns the resolution of the camera as (width, height). """
        pass


class StereoParamsInterface(ABC):
    """ Abstract interface for stereo calibration parameters. """

    class StereoCamera(Enum):
        LEFT = 'left'
        RIGHT = 'right'

    @abstractmethod
    def load_params(self, input_data):
        """ Load stereo calibration parameters from YAML file or string. """
        pass

    @abstractmethod
    def get_intrinsics(self):
        """ Return camera intrinsic matrices. """
        pass

    @abstractmethod
    def get_baseline(self):
        """ Return the stereo camera baseline. """
        pass

    @abstractmethod
    def get_z_max(self, d_min=1.0):
        """
        Return the maximum depth (Z_max) for which the disparity measurement is beneficial/reliable.
        Typically, Z_max is computed as (f_avg * baseline) / d_min,
        where d_min is the minimal measurable disparity.
        """
        pass


class StereoRectificationInterface(ABC):
    """ Abstract interface for stereo rectification. """

    @abstractmethod
    def rectify_images(self, img_left, img_right):
        """ Rectify stereo images using calibration parameters. """
        pass

    @abstractmethod
    def get_rectification_matrices(self):
        """ Return rectification matrices. """
        pass

    @abstractmethod
    def get_rectification_masks(self):
        """
        Returns precomputed rectification masks.

        Returns:
            tuple: (rectification_mask, roi_mask)
        """
        pass


class StereoDisparityInterface(ABC):
    """
    Abstract interface for stereo disparity computation.
    """

    @abstractmethod
    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """
        Computes the disparity map.

        Args:
            img_left (numpy.ndarray): Left input image.
            img_right (numpy.ndarray): Right input image.

        Returns:
            numpy.ndarray: Disparity map.
        """
        pass
