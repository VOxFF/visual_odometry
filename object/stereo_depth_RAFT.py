"""
https://github.com/princeton-vl/RAFT-Stereo
"""

import sys
sys.path.append("../external/RAFT-Stereo")

import os
import torch
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from abc import ABC, abstractmethod
from core.utils.utils import InputPadder
from core.raft_stereo import RAFTStereo

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")

DEPTH_MIN = 0
DEPTH_MAX = 20
DISP_MIN = 0
DISP_MAX = 40

class AbstractDisparitySolver(ABC):

    @abstractmethod
    def disparity(self, left_img, right_img):
        pass

class AbstractDepthSolver(ABC):
    def __init__(self, focal_length_px :float, base_line: float):
        self.f = focal_length_px
        self.B = base_line
    @abstractmethod
    def depth (self, left_img, right_img):
        pass
class DisparityRAFT(AbstractDisparitySolver):
    def __init__(self, args = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not args:
            self.args = argparse.Namespace(
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
        else:
            self.args = args

        self.modelDP = torch.nn.DataParallel(RAFTStereo(self.args), device_ids=[0])
        self.modelDP.load_state_dict(torch.load(checkpoint))

        self.model = self.modelDP.module
        self.model.to(self.device)
        self.model.eval()

    def disparity(self, left_img, right_img):
        with torch.no_grad():
            image1, image2 = self._loadImage(left_img), self._loadImage(right_img)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = self.model(image1, image2, iters=32, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

        return -flow_up.cpu().numpy().squeeze()

    def _loadImage(self, image_param):
        if isinstance(image_param, str):  # a file path
            img = Image.open(image_param)
        elif isinstance(image_param, np.ndarray):  # an OpenCV image
            img = Image.fromarray(cv2.cvtColor(image_param, cv2.COLOR_BGR2RGB))
        else:
            raise TypeError("Unsupported image format")

        # Convert grayscale images to RGB
        if img.mode in ("L", "LA"):
            img = img.convert("RGB")

        img = np.array(img).astype(np.uint8)  # to NumPy array
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # to Torch Tensor
        return img[None].to(self.device)

class DepthSimple(AbstractDepthSolver):
    def __init__(self, focal_length_px :float, base_line: float, disparity_solver):
        super().__init__(focal_length_px, base_line)
        self.disparity_solver = disparity_solver

    def depth(self, left_img, right_img):
        disparity_np = self.disparity_solver.disparity(left_img, right_img)
        import matplotlib.pyplot as plt

        # print(f'{np.min(disparity_np)} and {np.max(disparity_np)}')
        valid = disparity_np >= 0
        depth_metric = np.zeros_like(disparity_np, dtype=np.float32)
        depth_metric[valid] = (self.f * self.B) / abs(disparity_np[valid])
        #depth_metric = (f * B) / abs(disparity_np)
        depth_metric = np.clip(depth_metric, DEPTH_MIN, DEPTH_MAX)
        disparity_np = np.clip(depth_metric, DISP_MIN, DISP_MAX)
        return disparity_np, depth_metric

"""
here we go
"""
#checkpoint = '/home/roman/Rainbow/camera/models/raft/raftstereo-sceneflow.pth'
checkpoint = '/home/roman/Rainbow/camera/models/raft/raftstereo-eth3d.pth'
disparity_solver = DisparityRAFT()
depth_solver = DepthSimple(277.48, 0.07919, disparity_solver)


single_frame = False
path = '/home/roman/Downloads/fpv_datasets/outdoor_forward_1_snapdragon_with_gt/'
mask = cv2.imread('/home/roman/Downloads/fpv_datasets/mask.png', cv2.IMREAD_GRAYSCALE).astype(np.uint8) == 0

if single_frame:
    #img_idx = 1489
    img_idx = 2148
    #img_idx = 2375

    left_file = 'img/image_0_'+str(img_idx)+'.png'
    right_file = 'img/image_1_'+str(img_idx)+'.png'

    disparity_np, depth_metric = depth_solver.depth(path + left_file, path + right_file)

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

else:
    limit = 0 #200
    left_txt, right_txt = 'left_images.txt', 'right_images.txt'

    # Read the file, ignoring comments (#) in the header
    df = pd.read_csv(path + left_txt, delim_whitespace=True, comment="#", names=["id", "timestamp", "image_name"])
    left_files = df["image_name"].tolist()

    df = pd.read_csv(path + right_txt, delim_whitespace=True, comment="#", names=["id", "timestamp", "image_name"])
    right_files = df["image_name"].tolist()

    if limit:
        left_files = left_files[:limit]
        right_files = right_files[:limit]


    i = 0
    os.makedirs(os.path.dirname(path + "out_depth/"), exist_ok=True)
    os.makedirs(os.path.dirname(path + "out_disp/"), exist_ok=True)

    def normalize(img, min, max):
        # Normalize using global min/max
        normalized_disp = ((img - min) / (max - min) * 255).astype(np.uint8)
        return normalized_disp

    for left_file, right_file in zip(left_files, right_files):
        disparity_np, depth_metric = depth_solver.depth(path + left_file, path + right_file)
        disparity_np = normalize(disparity_np, DISP_MIN, DISP_MAX)
        depth_metric = normalize(depth_metric, DISP_MIN, DISP_MAX)


        parts = left_file.split("_")  # Split by '_'
        index = int(parts[-1].split(".")[0])  # last part before ".png"


        plt.imsave(path+"out_disp/" + str(index) + "_disparity.png", np.where(mask, disparity_np, 0), cmap="jet")
        plt.imsave(path+"out_depth/" + str(index) + "_depth.png", np.where(mask, depth_metric, 0), cmap="inferno")
        i += 1
        if i%20 == 0:
            print(f"{i} of {len(left_files)}")


