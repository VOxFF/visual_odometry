import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torchvision.transforms as transforms


# model = torch.hub.load("zhyever/DepthFormer", "depthformer_model",
#                          pretrained=True, trust_repo=True, force_reload=True)

import sys
sys.path.append("/home/roman/Downloads/Monocular-Depth-Estimation-Toolbox")

from depth.depthformer import DepthFormerModel

# Initialize your model as needed
model = DepthFormerModel(**your_kwargs)
checkpoint = torch.load("path_to_checkpoint.pth", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])

model.eval()


transform_depthformer = transforms.Compose([
    transforms.Resize((384, 384)),  # Example target size; update if necessary.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


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
img_pil = Image.fromarray(img_rgb)
input_tensor = transform_depthformer(img_pil).unsqueeze(0)

# --- Run inference ---
with torch.no_grad():
    # Assume the model outputs a single-channel relative depth map
    prediction = model(input_tensor)

# --- Resize prediction to the original image size ---
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img_rgb.shape[:2],
    mode="bicubic",
    align_corners=False
).squeeze()

depth_map = prediction.cpu().numpy()

# --- Plotting: Show the original image and the estimated depth map side by side ---
plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

plt.subplot(gs[0])
plt.imshow(img_rgb)
plt.title(right_file)
plt.axis("off")

plt.subplot(gs[1])
im = plt.imshow(depth_map, cmap="inferno")
plt.title("Estimated Depth using DepthFormer")
plt.axis("off")
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
