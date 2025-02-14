import cv2
import numpy as np
import pandas as pd
import glob

path = '/home/roman/Downloads/fpv_datasets/outdoor_forward_1_snapdragon_with_gt/'
output_video = 'output_video.mp4'
files_source = 'left_images.txt'

df = pd.read_csv(path + 'left_images.txt', delim_whitespace=True, comment="#", names=["id", "timestamp", "image_name"])
image_files = df["image_name"].tolist()


# Define video properties
print(path + image_files[0])
img = cv2.imread(path + image_files[0])
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
w,h = img_rgb.shape[1], img_rgb.shape[0]

frame_width = 3 * w  # 3 images side by side
frame_height = h
fps = 10  # Frames per second

# Video writer setup

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
video_writer = cv2.VideoWriter(path+output_video, fourcc, fps, (frame_width, frame_height))

# Process each frame
for i, image_file in enumerate(image_files):
    parts = image_file.split("_")  # Split by '_'
    index = int(parts[-1].split(".")[0])  # last part before ".png"
    img = cv2.imread(path + image_file)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    disp = cv2.imread(path + "out_disp/" + str(index) + "_disparity.png")
    depth = cv2.imread(path + "out_depth/" + str(index) + "_depth.png")


    # Ensure all images have the same height
    disp = cv2.resize(disp, (w, h))
    depth = cv2.resize(depth, (w, h))

    # Concatenate side-by-side
    combined_frame = np.hstack((img, disp, depth))

    # Write frame to video
    video_writer.write(combined_frame)

    if i % 20 == 0:
        print(f"{i} of {len(image_files)}")

# Release video writer
video_writer.release()
print(f"Video saved as {output_video}")
