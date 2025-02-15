import yaml
import numpy as np
import os
import json
from stereo_interfaces import StereoParamsInterface


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

            print(f"Loading YAML file: {input_data}")  # Debugging

            # Check if the file exists before opening
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"YAML file not found: {input_data}")

            with open(input_data, "r") as file:
                try:
                    data = yaml.safe_load(file)
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML file: {e}")

            # Pretty print the loaded YAML data
            print("Parsed YAML Data:")
            print(json.dumps(data, indent=4))  # Pretty print JSON-style output

        else:
            print("Input is not a file, assuming raw YAML string.")  # Debugging
            try:
                data = yaml.safe_load(input_data)
                print("Parsed YAML String:")
                print(json.dumps(data, indent=4))  # Pretty print JSON-style output
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
        self.baseline = np.linalg.norm(self.T)  # Baseline (magnitude of translation vector)

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
