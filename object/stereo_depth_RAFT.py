"""
https://github.com/princeton-vl/RAFT-Stereo
"""

import sys
sys.path.append("../external/RAFT-Stereo")

import torch
import cv2
import numpy as np
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from core.utils.utils import InputPadder
from core.raft_stereo import RAFTStereo

# Load your input image using OpenCV and convert it to RGB
checkpoint = '/home/roman/Rainbow/camera/models/raft/raftstereo-sceneflow.pth'
path = '/home/roman/Downloads/fpv_datasets/outdoor_forward_1_snapdragon_with_gt/img/'
mask = cv2.imread('/home/roman/Downloads/fpv_datasets/mask.png', cv2.IMREAD_GRAYSCALE).astype(np.uint8) == 0

#0-120
# left_file = 'image_0_1489.png'
# right_file = 'image_1_1489.png'

#0-300
left_file = 'image_0_2148.png'
right_file = 'image_1_2148.png'

# disp 0-400
# left_file = 'image_0_2375.png'
# right_file = 'image_1_2375.png'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cuda'

def load_image(imfile):
    img = Image.open(imfile)
    if img.mode in ("L", "LA"):
        img = img.convert("RGB")
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

args = argparse.Namespace(
    hidden_dims=[128, 128, 128],
    n_downsample=2,
    context_norm="batch",
    n_gru_layers=3,
    slow_fast_gru=False,
    corr_implementation="reg",
    corr_levels=4,
    corr_radius=4,
    shared_backbone=False,
    mixed_precision=False  # add if the code uses this flag
)

model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
model.load_state_dict(torch.load(checkpoint))

model = model.module
model.to(DEVICE)
model.eval()


with torch.no_grad():
    image1 = load_image(path+left_file)
    image2 = load_image(path+right_file)

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    _, flow_up = model(image1, image2, iters=32, test_mode=True)
    flow_up = padder.unpad(flow_up).squeeze()

disparity_np = -flow_up.cpu().numpy().squeeze()
#disparity_np = np.clip(disparity_np, 0.0, 120)

# Compute metric depth using calibration info: depth = f*B / disparity
f = 277.48  # example focal length in pixels
B = 0.07919  # example baseline in meters
valid = disparity_np > 1e-6
depth_metric = np.zeros_like(disparity_np, dtype=np.float32)
#depth_metric[valid] = (f * B) / abs(disparity_np[valid])
depth_metric = (f * B) / abs(disparity_np)
depth_metric = np.clip(depth_metric, 0.000, 40)

# Create a 2x2 plot:
# First row: left and right images.
# Second row: disparity map and metric depth map.
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].imshow(cv2.imread(path + left_file, cv2.IMREAD_GRAYSCALE), cmap='gray')
axs[0, 0].set_title("Left Image: " + left_file)
axs[0, 0].axis("off")

axs[0, 1].imshow(cv2.imread(path + right_file, cv2.IMREAD_GRAYSCALE), cmap='gray')
axs[0, 1].set_title("Right Image: " + right_file)
axs[0, 1].axis("off")

im_disp = axs[1, 0].imshow(np.where(mask, disparity_np, np.nan), cmap="jet")
axs[1, 0].set_title("Disparity Map")
axs[1, 0].axis("off")
fig.colorbar(im_disp, ax=axs[1, 0], fraction=0.046, pad=0.04)

im_depth = axs[1, 1].imshow(np.where(mask, depth_metric, np.nan), cmap="inferno")
axs[1, 1].set_title("Metric Depth Map")
axs[1, 1].axis("off")
fig.colorbar(im_depth, ax=axs[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
