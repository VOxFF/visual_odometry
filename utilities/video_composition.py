import cv2
import numpy as np
import os


def make_stacked_video(
        path: str,
        list_of_filenames: list,
        output_file: str,
        list_of_file_transformations: list = None,
        fps: int = 25
):
    """
    Creates a side-by-side stacked video from multiple sources with optional transformations.

    Args:
        path (str): Base path where images are stored.
        list_of_filenames (list): List of filenames for the main image sequence.
        output_file (str): Path to save the output video.
        list_of_file_transformations (list): List of lambda functions for filename transformations.
        fps (int): Frames per second for the video.
    """

    # Ensure at least the original images are included
    if list_of_file_transformations is None:
        list_of_file_transformations = [lambda x: x]

    # Read the first image to get dimensions
    first_img_path = os.path.join(path, list_of_filenames[0])
    first_img = cv2.imread(first_img_path)

    if first_img is None:
        raise ValueError(f"Error: Cannot read first image: {first_img_path}")

    h, w = first_img.shape[:2]
    frame_width = len(list_of_file_transformations) * w
    frame_height = h

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(os.path.join(path, output_file), fourcc, fps, (frame_width, frame_height))

    for i, filename in enumerate(list_of_filenames):
        transformed_images = []

        for transform in list_of_file_transformations:
            transformed_path = os.path.join(path, transform(filename))
            transformed_img = cv2.imread(transformed_path)

            if transformed_img is None:
                print(f"Warning: Missing image {transformed_path}, using black placeholder.")
                transformed_img = np.zeros((h, w, 3), dtype=np.uint8)

            transformed_img = cv2.resize(transformed_img, (w, h))
            transformed_images.append(transformed_img)

        # Stack images side by side
        combined_frame = np.hstack(transformed_images)

        # Write frame to video
        video_writer.write(combined_frame)

        if i % 20 == 0:
            print(f"Processed {i}/{len(list_of_filenames)} frames")

    # Release the writer
    video_writer.release()
    print(f"Video saved as {output_file}")

