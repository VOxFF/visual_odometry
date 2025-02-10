"""
https://github.com/princeton-vl/RAFT-Stereo
"""

import sys
sys.path.append("../external/RAFT-Stereo")

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from core.raft_stereo import RAFTStereo

import pkgutil

module_names = sorted([module.name for module in pkgutil.iter_modules()])
print(module_names)

#import raft_stereo


#from raft_stereo import RAFTStereo, autocast  # Import the model from the repository
# from raft_stereo import RAFTStereo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RAFTStereo()  # You might need to pass an args object here; adjust accordingly.
checkpoint = torch.load("/path/to/your/checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()

# Load your stereo images (for example, right image)
left_path = "/path/to/left_image.png"
right_path = "/path/to/right_image.png"
left_img = cv2.imread(left_path)
right_img = cv2.imread(right_path)
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

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(disparity_np, cmap="plasma")
plt.title("Disparity Map")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(depth_metric, cmap="inferno")
plt.title("Metric Depth Map")
plt.axis("off")
plt.show()
