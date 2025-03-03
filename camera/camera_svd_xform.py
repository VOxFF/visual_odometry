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



class CameraRansacXform(CameraXformInterface):
    """
    RANSAC-based implementation of CameraXformInterface to compute the rigid transformation
    (rotation and translation) between two sets of 3D points, robustly filtering out outliers.
    """

    def __init__(self, threshold: float = 0.05, iterations: int = 1000):
        """
        Initializes the RANSAC parameters.

        Args:
            threshold (float): Distance threshold to consider a correspondence as an inlier.
            iterations (int): Number of RANSAC iterations.
        """
        self.threshold = threshold
        self.iterations = iterations

    def compute_camera_xform(self, P: np.ndarray, Q: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Compute the rigid camera transformation that aligns a set of 3D points P to Q,
        using RANSAC to robustly filter out outlier correspondences.

        Args:
            P (np.ndarray): An array of shape (N, 3) containing 3D points from the current frame.
            Q (np.ndarray): An array of shape (N, 3) containing corresponding 3D points from the next frame.

        Returns:
            tuple:
                - R (np.ndarray): A 3x3 rotation matrix.
                - t (np.ndarray): A translation vector of length 3.
        """
        # Check that input arrays have the correct shape.
        if P.shape[0] != Q.shape[0] or P.shape[1] != 3 or Q.shape[1] != 3:
            raise ValueError("Input arrays must be of shape (N, 3)")
        N = P.shape[0]
        if N < 3:
            raise ValueError("At least 3 points are required for RANSAC.")

        best_inliers = None
        best_R = None
        best_t = None
        max_inliers_count = 0

        # RANSAC iterations
        for _ in range(self.iterations):
            # Randomly sample 3 distinct indices (minimal set for rigid estimation)
            indices = np.random.choice(N, 3, replace=False)
            P_sample = P[indices]
            Q_sample = Q[indices]

            # Compute candidate transformation using the Kabsch algorithm.
            centroid_P = np.mean(P_sample, axis=0)
            centroid_Q = np.mean(Q_sample, axis=0)
            P_centered = P_sample - centroid_P
            Q_centered = Q_sample - centroid_Q
            H = P_centered.T @ Q_centered
            U, S, Vt = np.linalg.svd(H)
            R_candidate = Vt.T @ U.T
            # Correct for reflection if necessary.
            if np.linalg.det(R_candidate) < 0:
                Vt[2, :] *= -1
                R_candidate = Vt.T @ U.T
            t_candidate = centroid_Q - R_candidate @ centroid_P

            # Apply the candidate transformation to all points.
            P_transformed = (R_candidate @ P.T).T + t_candidate
            errors = np.linalg.norm(P_transformed - Q, axis=1)
            inliers = errors < self.threshold
            inliers_count = np.sum(inliers)

            # Update best model if this iteration has more inliers.
            if inliers_count > max_inliers_count:
                max_inliers_count = inliers_count
                best_inliers = inliers
                best_R = R_candidate
                best_t = t_candidate

        # Recompute transformation using all inliers if enough were found.
        if max_inliers_count >= 3:
            P_inliers = P[best_inliers]
            Q_inliers = Q[best_inliers]
            centroid_P = np.mean(P_inliers, axis=0)
            centroid_Q = np.mean(Q_inliers, axis=0)
            P_centered = P_inliers - centroid_P
            Q_centered = Q_inliers - centroid_Q
            H = P_centered.T @ Q_centered
            U, S, Vt = np.linalg.svd(H)
            R_final = Vt.T @ U.T
            if np.linalg.det(R_final) < 0:
                Vt[2, :] *= -1
                R_final = Vt.T @ U.T
            t_final = centroid_Q - R_final @ centroid_P
        else:
            # Fallback if no sufficient inliers are found.
            R_final = np.eye(3)
            t_final = np.zeros(3)

        return R_final, t_final
