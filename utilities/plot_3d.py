# 3d_plot.py
import numpy as np
import matplotlib.pyplot as plt


class TrajectoryPlot:
    def __init__(self, global_positions, elevation=0, azimuth=90, zoom_distance=5,
                 small_font_size=8, axis_length_factor=0.1, display_axes_values=True, show_labels=False):
        """
        Initializes the TrajectoryPlot with a set of trajectories and view parameters.

        Args:
            global_positions: Either a numpy array of shape (N, 3) (for a single trajectory)
                              or a list of tuples/lists, where each tuple contains positions
                              from different sources (e.g. computed, ground truth, etc.) for each frame.
            elevation (float): Elevation angle (degrees) for the main 3D view.
            azimuth (float): Azimuth angle (degrees) for the main 3D view.
            zoom_distance (float): Zoom distance for the 3D view.
            small_font_size (int): Font size for labels and tick labels.
            axis_length_factor (float): Factor to scale the axis arrows relative to the overall trajectory range.
            display_axes_values (bool): If True, text labels for the fixed world axes are drawn.
            show_labels (bool): Flag for the fixed world axes triplet. (When True, labels are drawn.)
        """
        # Process input to build a list of trajectories.
        # If the input is a numpy array, assume it's a single trajectory.
        if isinstance(global_positions, np.ndarray):
            self.all_trajectories = [global_positions]
        elif isinstance(global_positions, list):
            if len(global_positions) == 0:
                self.all_trajectories = []
            else:
                # Check if each frame is a tuple/list (i.e. multiple trajectories per frame)
                if isinstance(global_positions[0], (list, tuple)):
                    num_traj = len(global_positions[0])
                    self.all_trajectories = []
                    for j in range(num_traj):
                        traj_j = []
                        for frame in global_positions:
                            elem = np.array(frame[j])
                            # If the element is not of length 3, pad or trim.
                            if elem.size < 3:
                                elem = np.pad(elem, (0, 3 - elem.size), mode='constant')
                            elif elem.size > 3:
                                elem = elem[:3]
                            traj_j.append(elem)
                        # Use vstack to get a (N, 3) array.
                        self.all_trajectories.append(np.vstack(traj_j))
                else:
                    # It's a list of positions; convert directly.
                    self.all_trajectories = [np.array(global_positions)]
        else:
            raise ValueError("global_positions must be a list or numpy array")

        # Compute overall extents using all trajectories.
        if len(self.all_trajectories) > 0:
            all_pos = np.concatenate(self.all_trajectories, axis=0)
            self.min_vals = all_pos.min(axis=0)
            self.max_vals = all_pos.max(axis=0)
        else:
            self.min_vals = self.max_vals = np.array([0, 0, 0])
        expand_view = 2  # add a small margin
        self.x_range = self.max_vals[0] - self.min_vals[0] + expand_view
        self.y_range = self.max_vals[1] - self.min_vals[1] + expand_view
        self.z_range = self.max_vals[2] - self.min_vals[2] + expand_view
        self.max_range = max(self.x_range, self.y_range, self.z_range)
        self.x_mid = (self.max_vals[0] + self.min_vals[0]) / 2
        self.y_mid = (self.max_vals[1] + self.min_vals[1]) / 2
        self.z_mid = (self.max_vals[2] + self.min_vals[2]) / 2

        # Define a fixed origin for drawing the world axes triplet.
        offset = self.max_range * 0.05
        self.world_axes_origin = np.array([
            self.min_vals[0] + offset,
            self.min_vals[1] + offset,
            self.min_vals[2] + offset
        ])

        self.elevation = elevation
        self.azimuth = azimuth
        self.zoom_distance = zoom_distance
        self.small_font_size = small_font_size
        self.axis_length_factor = axis_length_factor
        self.display_axes_values = display_axes_values
        self.show_labels = show_labels

        # Define a simple colormap for trajectories.
        self.colormap = ['blue', 'green', 'purple', 'orange', 'brown', 'magenta']

    def _plot_trajectory(self, ax, positions, color, start_marker_color=None, end_marker_color=None):
        """
        Draws a trajectory line with markers on the provided 3D axis.

        Args:
            ax: The 3D axis.
            positions (np.ndarray): Array of shape (N, 3) representing positions.
            color (str): Color of the line.
            start_marker_color (str): Color for the start marker (defaults to line color).
            end_marker_color (str): Color for the end marker (defaults to line color).
        """
        if start_marker_color is None:
            start_marker_color = color
        if end_marker_color is None:
            end_marker_color = color
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-', color=color)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color=start_marker_color, s=50)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color=end_marker_color, s=50)

    def _plot_triplet(self, ax, T, arrow_len=None, origin=None, show_labels=False):
        """
        Draws a coordinate axes triplet based on the transformation matrix T.

        Args:
            ax: The 3D axis.
            T (np.ndarray): A 4x4 transformation matrix (upper 3x3 is rotation).
            arrow_len (float, optional): Length of the arrows (if None, computed from max_range).
            origin (np.ndarray, optional): Origin for the triplet (defaults to T's translation).
            show_labels (bool): If True, draws text labels at the arrow tips.
        """
        if arrow_len is None:
            arrow_len = self.max_range * self.axis_length_factor
        if origin is None:
            origin = T[:3, 3]
        R = T[:3, :3]
        ax.quiver(origin[0], origin[1], origin[2],
                  arrow_len * R[0, 0], arrow_len * R[1, 0], arrow_len * R[2, 0],
                  color='r', linewidth=2)
        ax.quiver(origin[0], origin[1], origin[2],
                  arrow_len * R[0, 1], arrow_len * R[1, 1], arrow_len * R[2, 1],
                  color='g', linewidth=2)
        ax.quiver(origin[0], origin[1], origin[2],
                  arrow_len * R[0, 2], arrow_len * R[1, 2], arrow_len * R[2, 2],
                  color='b', linewidth=2)
        if show_labels:
            ax.text(origin[0] + arrow_len, origin[1], origin[2], 'X', color='r', fontsize=self.small_font_size)
            ax.text(origin[0], origin[1] + arrow_len, origin[2], 'Y', color='g', fontsize=self.small_font_size)
            ax.text(origin[0], origin[1], origin[2] + arrow_len, 'Z', color='b', fontsize=self.small_font_size)

    def plot(self, current_T, idx=None, trajectory=-1):
        """
        Creates and returns a 3D plot.

        Args:
            current_T (np.ndarray): A 4x4 transformation matrix representing the current camera pose.
            idx (int, optional): If provided, each trajectory is restricted to the first idx+1 points.
            trajectory (int):
                - If -1, plot all trajectories (each with its own color from the colormap).
                - Otherwise, plot only the trajectory with that index.

        Returns:
            fig: The generated matplotlib figure.
        """
        # Restrict each trajectory if idx is provided.
        if idx is not None:
            trajs = [traj[:idx + 1] for traj in self.all_trajectories]
        else:
            trajs = self.all_trajectories

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectories.
        if trajectory == -1:
            for i, traj in enumerate(trajs):
                col = self.colormap[i % len(self.colormap)]
                self._plot_trajectory(ax, np.asarray(traj), color=col)
        else:
            if trajectory < len(trajs):
                col = self.colormap[trajectory % len(self.colormap)]
                self._plot_trajectory(ax, np.asarray(trajs[trajectory]), color=col)
            else:
                print(f"Trajectory index {trajectory} is out of range; nothing plotted.")

        # Draw fixed world axes triplet using a precomputed transformation.
        T_world = np.eye(4)
        T_world[:3, 3] = self.world_axes_origin
        arrow_len = self.max_range * self.axis_length_factor
        # Draw world axes triplet with labels.
        self._plot_triplet(ax, T_world, arrow_len=arrow_len, show_labels=True)

        # Draw camera axes triplet at the current camera location (no labels).
        self._plot_triplet(ax, current_T, show_labels=False)

        # Set fixed axis limits.
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
