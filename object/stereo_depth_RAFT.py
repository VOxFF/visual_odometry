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
from core.raft_stereo import RAFTStereo

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

'''
#model = RAFTStereo(args)
model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
#checkpoint = torch.load("/home/roman/Rainbow/camera/models/raft/raftstereo-middlebury.pth", map_location=device)
#model.load_state_dict(checkpoint["state_dict"])
checkpoint = torch.load("/home/roman/Rainbow/camera/models/raft/raftstereo-middlebury.pth")
if not checkpoint:
    raise RuntimeError('no checkpoint')

model = model.module
model.load_state_dict(checkpoint)

'''

# Wrap the model in DataParallel (this will make the model's state dict keys be prefixed with "module.")
model = nn.DataParallel(RAFTStereo(args), device_ids=[0])
# checkpoint = torch.load("/home/roman/Rainbow/camera/models/raft/raftstereo-middlebury.pth", map_location=device)
# model.load_state_dict(checkpoint["state_dict"])  # or checkpoint, depending on your checkpoint structure

# Load the checkpoint, setting weights_only=True per the recommendation.
checkpoint = torch.load(
    "/home/roman/Rainbow/camera/models/raft/raftstereo-middlebury.pth",
    map_location=device,
    weights_only=True
)

# If the checkpoint is a dictionary with a "state_dict" key, use that;
# otherwise, assume the checkpoint itself is the state dictionary.
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)



model = model.module  # optionally, extract the underlying model if you don't need DataParallel further



model.to(device)
model.eval()



# Load your stereo images (for example, right image)
# Load your input image using OpenCV and convert it to RGB
path = '/home/roman/Downloads/fpv_datasets/outdoor_forward_1_snapdragon_with_gt/img/'
# left_file = 'image_0_1489.png'
# right_file = 'image_1_1489.png'

left_file = 'image_0_2148.png'
right_file = 'image_1_2148.png'

# left_file = 'image_0_2375.png'
# right_file = 'image_1_2375.png'

left_img = cv2.imread(path+left_file)
right_img = cv2.imread(path+right_file)
left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

# Preprocess images to tensor (you might need to adjust normalization/resizing)
def preprocess(img):
    img = cv2.resize(img, (640,480))
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
    return tensor

left_tensor = preprocess(left_img)
right_tensor = preprocess(right_img)

with torch.no_grad():
    output = model(left_tensor, right_tensor, iters=32, test_mode=True)
    disparity = output[-1] if isinstance(output, (list, tuple)) else output

disparity_np = disparity.squeeze().cpu().numpy()

# Compute metric depth using calibration info: depth = f*B / disparity
f = 277.48  # example focal length in pixels
B = 0.07919  # example baseline in meters
valid = disparity_np > 1e-6
depth_metric = np.zeros_like(disparity_np, dtype=np.float32)
depth_metric[valid] = (f * B) / disparity_np[valid]

# Create a 2x2 plot:
# First row: left and right images.
# Second row: disparity map and metric depth map.
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].imshow(left_img)
axs[0, 0].set_title("Left Image: " + left_file)
axs[0, 0].axis("off")

axs[0, 1].imshow(right_img)
axs[0, 1].set_title("Right Image: " + right_file)
axs[0, 1].axis("off")

im_disp = axs[1, 0].imshow(disparity_np, cmap="plasma")
axs[1, 0].set_title("Disparity Map")
axs[1, 0].axis("off")
fig.colorbar(im_disp, ax=axs[1, 0], fraction=0.046, pad=0.04)

im_depth = axs[1, 1].imshow(depth_metric, cmap="inferno")
axs[1, 1].set_title("Metric Depth Map")
axs[1, 1].axis("off")
fig.colorbar(im_depth, ax=axs[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
