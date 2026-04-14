import numpy as np
from abc import ABC, abstractmethod


class IMUInterface(ABC):
    """
    Abstract interface for IMU initialization.
    Provides the initial camera-to-world rotation at startup,
    which is used to express the VO trajectory in the world frame.
    """

    @abstractmethod
    def get_initial_pose(self) -> np.ndarray:
        """
        Returns the 3x3 rotation matrix that maps the camera optical frame
        into the world frame at t=0.

        On a real drone this comes from:
          - the camera-IMU extrinsic (calibrated offline)
          - the IMU's initial attitude reading in the world frame

        Returns:
            np.ndarray: 3x3 rotation matrix R_cam_in_world.
        """
