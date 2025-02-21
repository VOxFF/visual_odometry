"""
https://github.com/princeton-vl/RAFT-Stereo
"""


import torch
import cv2
import numpy as np
import torch.nn as nn
import argparse
from PIL import Image

from stereo_core.utils.utils import InputPadder
from stereo_core.raft_stereo import RAFTStereo
from stereo.stereo_interfaces import StereoDisparityInterface, StereoRectificationInterface

class DisparityRAFT(StereoDisparityInterface):
    def __init__(self, checkpoint: str, rectification: StereoRectificationInterface = None, iters: int = 32):
        """
        Initializes RAFT-Stereo-based disparity computation with optional rectification.

        Args:
            checkpoint (str): Path to the RAFT-Stereo model checkpoint.
            rectification (StereoRectificationInterface, optional): Rectification instance.
            iters (int): Number of iterations for the RAFT-Stereo algorithm.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iters = iters
        self.flow = None  # ✅ Stores flow for next frame propagation
        self.rectification = rectification  # ✅ Store rectification instance

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
            mixed_precision=False
        )

        self.modelDP = torch.nn.DataParallel(RAFTStereo(self.args), device_ids=[0])
        self.modelDP.load_state_dict(torch.load(checkpoint, map_location=self.device))

        self.model = self.modelDP.module
        self.model.to(self.device)
        self.model.eval()

    def compute_disparity(self, img_left, img_right) -> np.ndarray:
        """
        Computes the disparity map using RAFT-Stereo.

        Args:
            img_left (numpy.ndarray or str): Left input image.
            img_right (numpy.ndarray or str): Right input image.

        Returns:
            numpy.ndarray: Disparity map.
        """
        if img_left is None or img_right is None:
            raise ValueError("One or both images were not found or invalid.")

        # Apply rectification if available
        if self.rectification:
            img_left, img_right = self.rectification.rectify_images(img_left, img_right)

        # Convert images for RAFT-Stereo
        image1, image2 = self._loadImage(img_left), self._loadImage(img_right)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        # Ensure `self.flow` is correctly sized
        if self.flow is not None:
            H, W = image1.shape[2], image1.shape[3]
            self.flow = torch.zeros(1, 2, H // 4, W // 4, device=image1.device)
            self.flow = self.flow.to(image1.device)

        with torch.no_grad():
            _, flow_up = self.model(image1, image2, iters=self.iters, flow_init=self.flow, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

        self.flow = flow_up  # Store for next frame propagation
        return -flow_up.cpu().numpy().squeeze()

    def _loadImage(self, image_param):
        """Converts input image (file path or NumPy array) to the required format for RAFT-Stereo."""
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
