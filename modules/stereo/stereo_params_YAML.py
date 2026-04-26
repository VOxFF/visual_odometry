import yaml
import numpy as np
import os
import json
from modules.stereo.stereo_interfaces import CameraParametersInterface
from modules.stereo.stereo_interfaces import StereoParamsInterface

import numpy as np
from modules.stereo.stereo_interfaces import CameraParametersInterface


class CameraParameters(CameraParametersInterface):
    """
    Implementation of CameraParametersInterface for storing and retrieving camera intrinsic parameters.
    """

    def __init__(self, K: np.ndarray, D: np.ndarray, resolution: tuple):
        self.K = K
        self.D = D
        self.resolution = resolution

    def get_intrinsics(self) -> np.ndarray:
        return self.K

    def get_distortion_coeffs(self) -> np.ndarray:
        return self.D

    def get_resolution(self) -> tuple:
        return self.resolution

class StereoParamsYAML(StereoParamsInterface):
    """
    Implements StereoParamsInterface to load stereo calibration parameters from a YAML file.
    """

    def __init__(self, input_data):
        self.load_params(input_data)

    def load_params(self, input_data):
        """ Load and parse stereo calibration parameters from YAML file or text. """
        if isinstance(input_data, str) and os.path.isfile(input_data):
            if not input_data.lower().endswith((".yaml", ".yml")):
                raise ValueError("Invalid file format. Expected a .yaml or .yml file.")

            if not os.path.exists(input_data):
                raise FileNotFoundError(f"YAML file not found: {input_data}")

            with open(input_data, "r") as file:
                try:
                    data = yaml.safe_load(file)
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML file: {e}")

        else:
            try:
                data = yaml.safe_load(input_data)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML string: {e}")

        # Ensure the parsed data is valid
        if not isinstance(data, dict):
            raise TypeError("Parsed YAML data is not a dictionary. Check the YAML format.")

        if "cam0" not in data or "cam1" not in data:
            raise KeyError("Missing 'cam0' or 'cam1' keys in the YAML file.")

        # Extract parameters
        self.K_l = self._intrinsic_matrix(data["cam0"]["intrinsics"])
        self.D_l = np.array(data["cam0"]["distortion_coeffs"])
        self.K_r = self._intrinsic_matrix(data["cam1"]["intrinsics"])
        self.D_r = np.array(data["cam1"]["distortion_coeffs"])
        self.R = np.array(data["cam1"]["T_cn_cnm1"])[:3, :3]
        self.T = np.array(data["cam1"]["T_cn_cnm1"])[:3, 3]
        self.resolution = tuple(data["cam0"]["resolution"])
        self.focal_length_px = (self.K_l[0, 0] + self.K_r[0, 0]) / 2  # Average focal length
        # self.baseline = np.linalg.norm(self.T)  # Bug 2: norm(T) overestimates when T has Y/Z components
        self.baseline = abs(self.T[0])  # Baseline (horizontal separation after rectification)

    def _intrinsic_matrix(self, intrinsics):
        """ Helper function to convert intrinsics to a 3x3 matrix. """
        fx, fy, cx, cy = intrinsics
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def get_intrinsics(self):
        """ Returns left and right camera intrinsic matrices. """
        return {"K_l": self.K_l, "K_r": self.K_r}

    def get_baseline(self):
        """ Returns the stereo camera baseline. """
        return self.baseline

    def set_rectified_params(self, P1: np.ndarray, P2: np.ndarray):
        """
        Store the rectified intrinsic matrices (from StereoRectification).
        Must be called after rectification is computed so that get_camera_params()
        returns the correct K for back-projecting pixels in the rectified image.

        Args:
            P1 (np.ndarray): 3x4 rectified projection matrix for the left camera.
            P2 (np.ndarray): 3x4 rectified projection matrix for the right camera.
        """
        self.K_l_rect = P1[:3, :3]
        self.K_r_rect = P2[:3, :3]

    def get_camera_params(self, camera: "StereoParamsInterface.StereoCamera"):
        """
        Returns the camera parameters for the specified camera.
        Returns rectified intrinsics if set_rectified_params() has been called,
        otherwise falls back to the original (pre-rectification) intrinsics.

        Args:
            camera (StereoParamsInterface.StereoCamera): The camera side (LEFT or RIGHT).

        Returns:
            CameraParameters: An instance containing the parameters of the selected camera.
        """
        if camera == StereoParamsInterface.StereoCamera.LEFT:
            # Bug 1 fix: use rectified K from P1 instead of original K_l
            K = getattr(self, 'K_l_rect', self.K_l)
            # K = self.K_l  # Bug 1: original pre-rectification intrinsics
            return CameraParameters(K, self.D_l, self.resolution)
        elif camera == StereoParamsInterface.StereoCamera.RIGHT:
            K = getattr(self, 'K_r_rect', self.K_r)
            # K = self.K_r  # Bug 1: original pre-rectification intrinsics
            return CameraParameters(K, self.D_r, self.resolution)
        else:
            raise ValueError(f"Invalid camera side: {camera}")

    def get_z_max(self, d_min=1.0):
        """
        Return the maximum depth (Z_max) for which disparity measurement is beneficial.

        Assumes that a disparity below d_min (in pixels) is not reliable.
        Uses the formula:
            Z_max = (f_avg * baseline) / d_min
        where f_avg is the average focal length in pixels and baseline is the stereo baseline in meters.

        Args:
            d_min (float): Minimal measurable disparity in pixels (default is 1.0).

        Returns:
            float: The maximum reliable depth (in meters).
        """
        return self.focal_length_px * self.baseline / d_min
