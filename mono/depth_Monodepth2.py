"""
This model appears too old, dated 2019
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torchvision.transforms as transforms

# Load the pre-trained Monodepth2 model from PyTorch Hub
model = torch.hub.load("nianticlabs/monodepth2", "mono_640x192", pretrained=True, trust_repo=True)
model.eval()

# Define the preprocessing transform for Monodepth2
transform_monodepth = transforms.Compose([
    transforms.Resize((192, 640)),  # Resize to the model's expected input dimensions (height, width)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load your input image using OpenCV and convert it to RGB
path = '/home/roman/Downloads/fpv_datasets/outdoor_forward_1_snapdragon_with_gt/img/'
# left_file = 'image_0_1489.png'
# right_file = 'image_1_1489.png'

left_file = 'image_0_2148.png'
right_file = 'image_1_2148.png'

# left_file = 'image_0_2375.png'
# right_file = 'image_1_2375.png'

# For this example, we'll use the right image
img = cv2.imread(path + right_file)
if img is None:
    raise ValueError("Image not found.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the image to PIL format (Monodepth2 transform expects a PIL image)
img_pil = Image.fromarray(img_rgb)

# Preprocess the image
input_batch = transform_monodepth(img_pil).unsqueeze(0)

# Run inference to get the disparity map (relative depth)
with torch.no_grad():
    depth_estimation = model(input_batch)
    # The model returns a dictionary; key ("disp", 0) contains the predicted disparity
    disparity = depth_estimation[("disp", 0)].squeeze().cpu().numpy()

# Here, disparity serves as a relative depth map.
depth_map = disparity

# --- Plotting: Display the original image and the estimated depth map side by side ---
plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

plt.subplot(gs[0])
plt.imshow(img_rgb)
plt.title(right_file)
plt.axis("off")

plt.subplot(gs[1])
im = plt.imshow(depth_map, cmap="inferno")
plt.title("Estimated Depth (Disparity) using Monodepth2")
plt.axis("off")
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
