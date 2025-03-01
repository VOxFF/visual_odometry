# 3d_plot.py
import numpy as np
import matplotlib.pyplot as plt

class TrajectoryPlot:
    def __init__(self, global_positions, elevation=0, azimuth=90, zoom_distance=5,
                 small_font_size=8, axis_length_factor=0.1, display_axes_values=True):
        """
        Initializes the TrajectoryPlot with a set of global positions and view parameters.

        Args:
            global_positions (np.ndarray): Array of shape (N, 3) containing the world camera positions.
            elevation (float): Elevation angle (degrees) for the main 3D view.
            azimuth (float): Azimuth angle (degrees) for the main 3D view.
            zoom_distance (float): Zoom distance for the main 3D view.
            small_font_size (int): Font size for labels and tick labels.
            axis_length_factor (float): Factor to scale the camera/world axis arrows relative to the overall trajectory range.
            display_axes_values (bool): If True, text labels for the world axes are displayed.
        """
        self.global_positions = np.asarray(global_positions)
        self.elevation = elevation
        self.azimuth = azimuth
        self.zoom_distance = zoom_distance
        self.small_font_size = small_font_size
        self.axis_length_factor = axis_length_factor
        self.display_axes_values = display_axes_values

        # Precompute trajectory extents.
        self.min_vals = self.global_positions.min(axis=0)
        self.max_vals = self.global_positions.max(axis=0)
        expand_view = 2  # add a small margin
        self.x_range = self.max_vals[0] - self.min_vals[0] + expand_view
        self.y_range = self.max_vals[1] - self.min_vals[1] + expand_view
        self.z_range = self.max_vals[2] - self.min_vals[2] + expand_view
        self.max_range = max(self.x_range, self.y_range, self.z_range)
        self.x_mid = (self.max_vals[0] + self.min_vals[0]) / 2
        self.y_mid = (self.max_vals[1] + self.min_vals[1]) / 2
        self.z_mid = (self.max_vals[2] + self.min_vals[2]) / 2

        # Define a fixed origin for drawing world axes.
        # Here we choose the lower corner of the trajectory box with a small offset.
        offset = self.max_range * 0.05
        self.world_axes_origin = np.array([
            self.min_vals[0] + offset,
            self.min_vals[1] + offset,
            self.min_vals[2] + offset
        ])

    def plot(self, current_T, idx=None):
        """
        Creates and returns a 3D plot of the trajectory up to a given index (if provided)
        with a world axes triplet drawn at a fixed location and the camera axes drawn at the current camera location.

        Args:
            current_T (np.ndarray): Current 4x4 global transformation matrix.
                The upper 3x3 block is assumed to be the rotation.
            idx (int, optional): If provided, only the trajectory up to global_positions[:idx+1] is plotted;
                                 otherwise, the entire trajectory is used.

        Returns:
            fig (matplotlib.figure.Figure): The generated figure.
        """
        # Use the partial trajectory if idx is provided.
        if idx is None:
            traj = self.global_positions
        else:
            traj = self.global_positions[:idx + 1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory line.
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', color='blue')
        # Plot the starting point (green) and current camera position (red).
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=50)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=50)

        # 1. Draw world axes triplet (fixed and axis-aligned) at a fixed location.
        arrow_len = self.max_range * self.axis_length_factor
        origin = self.world_axes_origin
        ax.quiver(origin[0], origin[1], origin[2],
                  arrow_len, 0, 0, color='r', linewidth=2)
        ax.quiver(origin[0], origin[1], origin[2],
                  0, arrow_len, 0, color='g', linewidth=2)
        ax.quiver(origin[0], origin[1], origin[2],
                  0, 0, arrow_len, color='b', linewidth=2)
        if self.display_axes_values:
            ax.text(origin[0] + arrow_len, origin[1], origin[2], 'X', color='r', fontsize=self.small_font_size)
            ax.text(origin[0], origin[1] + arrow_len, origin[2], 'Y', color='g', fontsize=self.small_font_size)
            ax.text(origin[0], origin[1], origin[2] + arrow_len, 'Z', color='b', fontsize=self.small_font_size)

        # 2. Draw camera coordinate axes at the current camera location.
        cam_pos = current_T[:3, 3].copy()
        R_global = current_T[:3, :3]
        cam_arrow_len = self.max_range * self.axis_length_factor
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                  cam_arrow_len * R_global[0, 0], cam_arrow_len * R_global[1, 0], cam_arrow_len * R_global[2, 0],
                  color='r', linewidth=2)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                  cam_arrow_len * R_global[0, 1], cam_arrow_len * R_global[1, 1], cam_arrow_len * R_global[2, 1],
                  color='g', linewidth=2)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                  cam_arrow_len * R_global[0, 2], cam_arrow_len * R_global[1, 2], cam_arrow_len * R_global[2, 2],
                  color='b', linewidth=2)

        # Set fixed axis limits using precomputed extents.
        ax.set_xlim(self.x_mid - self.max_range / 2, self.x_mid + self.max_range / 2)
        ax.set_ylim(self.y_mid - self.max_range / 2, self.y_mid + self.max_range / 2)
        ax.set_zlim(self.z_mid - self.max_range / 2, self.z_mid + self.max_range / 2)
        ax.set_xlabel("X", fontsize=self.small_font_size)
        ax.set_ylabel("Y", fontsize=self.small_font_size)
        ax.set_zlabel("Z", fontsize=self.small_font_size)
        ax.tick_params(labelsize=self.small_font_size)
        ax.view_init(elev=self.elevation, azim=self.azimuth)
        ax.dist = self.zoom_distance

        return fig
