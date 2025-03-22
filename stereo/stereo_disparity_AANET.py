"""
uses https://github.com/VOxFF/aanet
the fork of https://github.com/haofeixu/aanet
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np

from stereo.stereo_interfaces import StereoDisparityInterface
from stereo.stereo_rectification import StereoRectificationInterface


# Import the AANet modules from the external submodule (assumed to be in /external)
#import external.aanet.nets as nets
import external.aanet.nets as nets
from external.aanet.dataloader import transforms
from external.aanet.utils import utils
from external.aanet.utils.file_io import read_img

class DisparityAANet(StereoDisparityInterface):
    """
    Implementation of StereoDisparityInterface using AANet.
    """

    def __init__(self, checkpoint: str, rectification: StereoRectificationInterface = None, max_disp: int = 192, **kwargs):
        """
        Initializes AANet-based stereo disparity computation.

        Args:
            checkpoint (str): Path to the pretrained AANet model.
            rectification (StereoRectificationInterface, optional): Rectification instance.
            max_disp (int): Maximum disparity value.
            **kwargs: Additional parameters for AANet.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rectification = rectification
        self.max_disp = max_disp

        # AANet parameters (with defaults that match the sample predict.py)
        self.num_downsample = kwargs.get('num_downsample', 2)
        self.feature_type = kwargs.get('feature_type', 'aanet')
        self.no_feature_mdconv = kwargs.get('no_feature_mdconv', False)
        self.feature_pyramid = kwargs.get('feature_pyramid', False)
        self.feature_pyramid_network = kwargs.get('feature_pyramid_network', False)
        self.feature_similarity = kwargs.get('feature_similarity', 'correlation')
        self.aggregation_type = kwargs.get('aggregation_type', 'adaptive')
        self.num_scales = kwargs.get('num_scales', 3)
        self.num_fusions = kwargs.get('num_fusions', 6)
        self.num_stage_blocks = kwargs.get('num_stage_blocks', 1)
        self.num_deform_blocks = kwargs.get('num_deform_blocks', 3)
        self.no_intermediate_supervision = kwargs.get('no_intermediate_supervision', False)
        self.refinement_type = kwargs.get('refinement_type', 'stereodrnet')
        self.mdconv_dilation = kwargs.get('mdconv_dilation', 2)
        self.deformable_groups = kwargs.get('deformable_groups', 2)

        # Initialize the AANet model.
        self.model = nets.AANet(
            self.max_disp,
            num_downsample=self.num_downsample,
            feature_type=self.feature_type,
            no_feature_mdconv=self.no_feature_mdconv,
            feature_pyramid=self.feature_pyramid,
            feature_pyramid_network=self.feature_pyramid_network,
            feature_similarity=self.feature_similarity,
            aggregation_type=self.aggregation_type,
            num_scales=self.num_scales,
            num_fusions=self.num_fusions,
            num_stage_blocks=self.num_stage_blocks,
            num_deform_blocks=self.num_deform_blocks,
            no_intermediate_supervision=self.no_intermediate_supervision,
            refinement_type=self.refinement_type,
            mdconv_dilation=self.mdconv_dilation,
            deformable_groups=self.deformable_groups
        ).to(self.device)

        # Load pretrained weights if provided.
        if os.path.exists(checkpoint):
            print("=> Loading pretrained AANet:", checkpoint)
            utils.load_pretrained_net(self.model, checkpoint, no_strict=True)
        else:
            print("=> Using random initialization for AANet")

        # Use DataParallel if multiple GPUs are available.
        if torch.cuda.device_count() > 1:
            print("=> Using {} GPUs".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)

        self.model.eval()

        # Define test-time image transform (to tensor and normalization).
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def compute_disparity(self, img_left, img_right) -> np.ndarray:
        """
        Computes the disparity map using AANet.

        Args:
            img_left (numpy.ndarray or str): Left input image.
            img_right (numpy.ndarray or str): Right input image.

        Returns:
            numpy.ndarray: Disparity map.
        """
        # First, load the images if file paths are provided.
        if isinstance(img_left, str):
            img_left = read_img(img_left)
        if isinstance(img_right, str):
            img_right = read_img(img_right)

        # Then apply rectification if available.
        if self.rectification:
            img_left, img_right = self.rectification.rectify_images(img_left, img_right)

        # Create a sample dictionary and apply transformation.
        sample = {'left': img_left, 'right': img_right}

        if sample['left'].ndim == 2:
            # Convert grayscale image (H, W) to 3-channel image (H, W, 3)
            sample['left'] = np.stack([sample['left']] * 3, axis=2)

        if sample['right'].ndim == 2:
            # Convert grayscale image (H, W) to 3-channel image (H, W, 3)
            sample['right'] = np.stack([sample['right']] * 3, axis=2)

        sample = self.test_transform(sample)

        # Convert to device and add batch dimension.
        left_tensor = sample['left'].to(self.device).unsqueeze(0)
        right_tensor = sample['right'].to(self.device).unsqueeze(0)

        # Get original image dimensions.
        ori_height, ori_width = left_tensor.shape[2], left_tensor.shape[3]

        # Determine padding factor.
        factor = 48 if self.refinement_type != 'hourglass' else 96
        new_height = math.ceil(ori_height / factor) * factor
        new_width = math.ceil(ori_width / factor) * factor

        pad_top = new_height - ori_height if ori_height < new_height else 0
        pad_right = new_width - ori_width if ori_width < new_width else 0

        # Pad images if necessary.
        if pad_top > 0 or pad_right > 0:
            left_tensor = F.pad(left_tensor, (0, pad_right, pad_top, 0))
            right_tensor = F.pad(right_tensor, (0, pad_right, pad_top, 0))

        # Run inference.
        with torch.no_grad():
            outputs = self.model(left_tensor, right_tensor)
            # Use the last output as the final disparity prediction.
            pred_disp = outputs[-1]

        # Upsample if necessary.
        if pred_disp.size(-1) < left_tensor.size(-1):
            pred_disp = pred_disp.unsqueeze(1)
            pred_disp = F.interpolate(pred_disp, size=(left_tensor.size(-2), left_tensor.size(-1)),
                                        mode='bilinear', align_corners=False) * (left_tensor.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)

        # Crop the disparity map back to the original size.
        if pad_top > 0 or pad_right > 0:
            pred_disp = pred_disp[:, pad_top:, :ori_width]

        # Return the disparity map as a NumPy array.
        disp = pred_disp[0].detach().cpu().numpy()
        return disp
