"""
https://github.com/princeton-vl/RAFT
"""

import torch
import cv2
import numpy as np
import torch.nn as nn
import argparse
from PIL import Image

from core.utils.utils import InputPadder
from core.raft import RAFT  # Optical Flow RAFT
from flow.flow_interfaces import OpticalFlowInterface


class OpticalFlowRAFT(OpticalFlowInterface):
    def __init__(self, checkpoint: str, iters: int = 32):
        """
        Initializes RAFT-based optical flow computation.

        Args:
            checkpoint (str): Path to the RAFT model checkpoint.
            iters (int): Number of iterations for the RAFT algorithm.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iters = iters
        self.flow = None  # Stores flow for next frame propagation

        # Define RAFT arguments
        self.args = argparse.Namespace(
            small=False,  # Standard RAFT model (not "small" version)
            mixed_precision=False,
            alternate_corr=False,
        )

        self.modelDP = torch.nn.DataParallel(RAFT(self.args), device_ids=[0])
        self.modelDP.load_state_dict(torch.load(checkpoint, map_location=self.device))

        self.model = self.modelDP.module
        self.model.to(self.device)
        self.model.eval()

    def compute_flow(self, img1, img2) -> np.ndarray:
        """
        Computes the optical flow between two images.

        Args:
            img1 (numpy.ndarray or str): First input image.
            img2 (numpy.ndarray or str): Second input image.

        Returns:
            numpy.ndarray: Optical flow map (dx, dy) for each pixel.
        """
        if img1 is None or img2 is None:
            raise ValueError("One or both images were not found or invalid.")

        # Convert images for RAFT
        image1, image2 = self._loadImage(img1), self._loadImage(img2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        # Ensure `self.flow` is correctly sized
        if self.flow is not None:
            H, W = image1.shape[2], image1.shape[3]
            self.flow = torch.zeros(1, 2, H // 4, W // 4, device=image1.device)
            self.flow = self.flow.to(image1.device)

        with torch.no_grad():
            _, flow_up = self.model(image1, image2, iters=self.iters, flow_init=self.flow, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

        self.flow = flow_up  # ✅ Store for next frame propagation
        return flow_up.cpu().numpy()

    def _loadImage(self, image_param):
        """Converts input image (file path or NumPy array) to the required format for RAFT."""
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
