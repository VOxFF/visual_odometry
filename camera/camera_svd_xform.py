from camera.camera_interfaces import CameraXformInterface
import numpy as np

class CameraSvdXform(CameraXformInterface):
    """
    Concrete implementation of CameraXformInterface using an SVD-based method (Kabsch algorithm)
    to compute the rigid camera transformation between two sets of 3D points.
    The results are in camera space, then adjusted to the drone (or IMU) center using a provided offset.
    """
    def __init__(self, offset=np.array([0.0, 0.0, 0.0])):
        """
        Args:
            offset (np.ndarray): A 3-element vector representing the camera's offset
                                 (in the camera coordinate system) relative to the drone's center.
        """
        self.offset = np.array(offset)

    def compute_camera_xform(self, P: np.ndarray, Q: np.ndarray) -> (np.ndarray, np.ndarray):
        # Compute centroids of both point sets.
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)

        # Center the points.
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # Compute covariance matrix.
        H = P_centered.T @ Q_centered

        # Perform SVD.
        U, S, Vt = np.linalg.svd(H)

        # Compute the rotation matrix.
        R = Vt.T @ U.T

        # Ensure a proper rotation (no reflection).
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        # Compute the translation vector in camera space.
        t = centroid_Q - R @ centroid_P

        # Adjust translation for the known camera offset.
        # This converts the camera pose into the drone/IMU center pose.
        t_corrected = t + R @ self.offset

        return R, t_corrected
