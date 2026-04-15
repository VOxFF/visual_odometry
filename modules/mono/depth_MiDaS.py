import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load the MiDaS model (choose model type: "DPT_Large" or "MiDaS_small")
#model_type = "DPT_Large"
model_type = "DPT_Hybrid"
#model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# Load the appropriate transform to preprocess input images
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# Load your input image using OpenCV and convert it to RGB
path = '/home/roman/Downloads/fpv_datasets/outdoor_forward_1_snapdragon_with_gt/img/'
# left_file = 'image_0_1489.png'
# right_file = 'image_1_1489.png'

left_file = 'image_0_2148.png'
right_file = 'image_1_2148.png'

# left_file = 'image_0_2375.png'
# right_file = 'image_1_2375.png'


img = cv2.imread(path + right_file)
if img is None:
    raise ValueError("Image not found.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocess the image
input_batch = transform(img_rgb)
# Run inference
with torch.no_grad():
    prediction = midas(input_batch)

# Resize prediction to the original image size
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img_rgb.shape[:2],
    mode="bicubic",
    align_corners=False
).squeeze()

# Convert to numpy array for visualization
depth_map = prediction.cpu().numpy()

# Display the estimated depth

plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

plt.subplot(gs[0])
plt.imshow(img)
plt.title(right_file)
plt.axis("off")

plt.subplot(gs[1])
plt.imshow(depth_map, cmap="inferno")
plt.title("Estimated Depth using MiDaS")
plt.axis("off")
plt.colorbar()

plt.show()

