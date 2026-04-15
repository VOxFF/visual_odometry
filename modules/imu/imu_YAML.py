import yaml
import numpy as np
import pandas as pd

from modules.imu.imu_interfaces import IMUInterface


class YamlIMU(IMUInterface):
    """
    IMU initializer that derives the initial camera-to-world rotation from:
      1. The camera-IMU extrinsic (T_cam_imu) stored in a Kalibr-format YAML file.
      2. The body orientation in the world frame at t=0, read from a ground truth file
         as a proxy for the IMU's initial attitude reading.

    On a real drone, step 2 would be replaced by a live IMU attitude estimate.
    """

    def __init__(self, yaml_file: str, image_list_path: str, gt_file_path: str):
        """
        Args:
            yaml_file:        Path to the Kalibr camera-IMU calibration YAML.
            image_list_path:  Path to the image list file (used to find the first image timestamp).
            gt_file_path:     Path to the ground truth file (timestamp tx ty tz qx qy qz qw).
        """
        self._yaml_file = yaml_file
        self._image_list_path = image_list_path
        self._gt_file_path = gt_file_path

    def get_initial_pose(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: 3x3 R_cam_in_world = R_body_in_world(q0) @ R_cam_imu.T
        """
        R_cam_imu = self._read_R_cam_imu()
        R_body_world = self._read_initial_body_orientation()
        return R_body_world @ R_cam_imu.T

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_R_cam_imu(self) -> np.ndarray:
        """Reads the 3x3 camera-to-IMU rotation from the YAML calibration file."""
        with open(self._yaml_file, 'r') as f:
            calib = yaml.safe_load(f)
        T_cam_imu = np.array(calib['cam0']['T_cam_imu'])
        return T_cam_imu[:3, :3]

    def _read_initial_body_orientation(self) -> np.ndarray:
        """
        Finds the GT entry closest to the first image timestamp and converts
        its quaternion to a 3x3 rotation matrix (body frame in world frame).
        """
        df_gt = pd.read_csv(self._gt_file_path, delim_whitespace=True, comment='#',
                            names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        df_gt.sort_values('timestamp', inplace=True)

        df_img = pd.read_csv(self._image_list_path, delim_whitespace=True, comment='#',
                             names=['id', 'timestamp', 'image_name'])
        first_img_ts = float(df_img.iloc[0]['timestamp'])

        first_gt_row = df_gt.loc[(df_gt['timestamp'] - first_img_ts).abs().idxmin()]
        return self._quaternion_to_rotation_matrix(
            first_gt_row['qx'], first_gt_row['qy'],
            first_gt_row['qz'], first_gt_row['qw']
        )

    @staticmethod
    def _quaternion_to_rotation_matrix(qx, qy, qz, qw) -> np.ndarray:
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        return np.array([
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),    2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),    1 - 2*(qx**2 + qy**2)]
        ])
