import cv2
import numpy as np
import os

def make_stacked_video(
        path: str,
        list_of_filenames: list,
        output_file: str,
        list_of_file_transformations: list = None,
        fps: int = 25,
        grid_shape: tuple = None,  # new parameter: (rows, cols)
        list_of_labels: list = None  # new optional parameter for labels per grid cell
):
    """
    Creates a grid-stacked video from multiple sources with optional transformations and optional text labels.

    Args:
        path (str): Base path where images are stored.
        list_of_filenames (list): List of filenames for the main image sequence.
        output_file (str): Path to save the output video.
        list_of_file_transformations (list): List of lambda functions for filename transformations.
        fps (int): Frames per second for the video.
        grid_shape (tuple): A tuple (rows, cols) defining the grid layout. If None, defaults to a single row.
        list_of_labels (list): Optional list of labels (strings) for each grid cell. If an element is None, no label is drawn.
                              If not provided, no labels are drawn.
    """

    # Ensure at least the original images are included
    if list_of_file_transformations is None:
        list_of_file_transformations = [lambda x: x]

    num_transformations = len(list_of_file_transformations)
    # Set grid layout; default is one row (i.e. (1, num_transformations))
    if grid_shape is None:
        rows = 1
        cols = num_transformations
    else:
        rows, cols = grid_shape

    total_slots = rows * cols
    # Adjust transformation list to match total grid slots.
    if total_slots < num_transformations:
        print(f"Warning: grid_shape {grid_shape} has fewer slots than transformations ({num_transformations}). Using only the first {total_slots} transformations.")
        list_of_file_transformations = list_of_file_transformations[:total_slots]
        num_transformations = total_slots
    elif total_slots > num_transformations:
        list_of_file_transformations.extend([lambda x: x] * (total_slots - num_transformations))
        num_transformations = total_slots

    # Adjust labels list similarly.
    if list_of_labels is None:
        labels = [None] * total_slots
    else:
        if len(list_of_labels) < total_slots:
            labels = list_of_labels + [None] * (total_slots - len(list_of_labels))
        elif len(list_of_labels) > total_slots:
            print(f"Warning: More labels than grid cells ({len(list_of_labels)} > {total_slots}). Using only the first {total_slots} labels.")
            labels = list_of_labels[:total_slots]
        else:
            labels = list_of_labels

    # Read the first image to get dimensions
    first_img_path = os.path.join(path, list_of_filenames[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        raise ValueError(f"Error: Cannot read first image: {first_img_path}")

    h, w = first_img.shape[:2]
    frame_width = cols * w
    frame_height = rows * h

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(os.path.join(path, output_file), fourcc, fps, (frame_width, frame_height))

    # Prepare a black image for placeholders
    black_placeholder = np.zeros((h, w, 3), dtype=np.uint8)

    for i, filename in enumerate(list_of_filenames):
        transformed_images = []
        for idx, transform in enumerate(list_of_file_transformations):
            transformed_path = os.path.join(path, transform(filename))
            transformed_img = cv2.imread(transformed_path)
            if transformed_img is None:
                print(f"Warning: Missing image {transformed_path}, using black placeholder.")
                transformed_img = black_placeholder.copy()
            else:
                transformed_img = cv2.resize(transformed_img, (w, h))
            # Add label if provided, using a smaller font (scale 0.5, thickness 1)
            if labels[idx] is not None:
                cv2.putText(transformed_img, labels[idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
            transformed_images.append(transformed_img)

        # Arrange images into the specified grid layout
        rows_images = []
        for r in range(rows):
            row_imgs = []
            for c in range(cols):
                idx = r * cols + c
                row_imgs.append(transformed_images[idx])
            row_combined = np.hstack(row_imgs)
            rows_images.append(row_combined)
        combined_frame = np.vstack(rows_images)

        video_writer.write(combined_frame)

        if i % 20 == 0:
            print(f"Processed {i}/{len(list_of_filenames)} frames")

    video_writer.release()
    print(f"Video saved as {output_file}")
