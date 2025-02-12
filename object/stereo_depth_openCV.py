import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths for left and right images (adjust accordingly)
path = '/home/roman/Downloads/fpv_datasets/outdoor_forward_1_snapdragon_with_gt/img/'
# left_file = 'image_0_1489.png'
# right_file = 'image_1_1489.png'

left_file = 'image_0_2148.png'
right_file = 'image_1_2148.png'

# left_file = 'image_0_2375.png'
# right_file = 'image_1_2375.png'



# Load images in grayscale (stereo matching works on intensity images)
img_left = cv2.imread(path + left_file, cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread(path + right_file, cv2.IMREAD_GRAYSCALE)

if img_left is None or img_right is None:
    raise ValueError("One or both images were not found. Check your file paths.")



# Compute disparity using StereoBM (you can also try StereoSGBM)
# numDisparities = 48   # must be divisible by 16
# blockSize = 15
# stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

# min_disp = 0
# num_disp = 32   # must be divisible by 16
# blockSize = 11
# stereo = cv2.StereoSGBM_create(
#     minDisparity=min_disp,
#     numDisparities=num_disp,
#     blockSize=blockSize,
#     P1=8 * 3 * blockSize**2,
#     P2=32 * 3 * blockSize**2,
#     disp12MaxDiff= 2, #1,
#     uniquenessRatio= 4, #10,
#     speckleWindowSize= 10, #100,
#     speckleRange= 4 #32
# )

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,  # try increasing this if needed
    blockSize=11,
    P1=8 * 1 * 5**2,
    P2=32 * 1 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=1,
    speckleWindowSize=100,
    speckleRange=16,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)


disparity = stereo.compute(img_left, img_right)
disparity = np.float32(disparity) / 16.0


# Camera calibration parameters from your YAML for cam0
# intrinsics: [fx, fy, cx, cy]
fx, fy, cx, cy = 277.4786896484645, 277.42548548840034, 320.1052053576385, 242.10083077857894

# Baseline from extrinsics for cam1 (x-translation, assuming horizontal alignment)
B = 0.07919358  # in meters

# Create a mask for pixels with positive disparity
mask = disparity > 0

# Initialize the depth map with zeros (same shape as disparity)
depth = np.zeros_like(disparity, dtype=np.float32)

# Compute depth only where disparity is valid
depth[mask] = (fx * B) / disparity[mask]


# --- two rows with two columns ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# First row: Original grayscale images
axs[0, 0].imshow(img_left, cmap='gray')
axs[0, 0].set_title("Left Image (Grayscale)")
axs[0, 0].axis("off")

axs[0, 1].imshow(img_right, cmap='gray')
axs[0, 1].set_title("Right Image (Grayscale)")
axs[0, 1].axis("off")

# Second row: Computed disparity and depth maps
im_disp = axs[1, 0].imshow(disparity, cmap='plasma')
axs[1, 0].set_title("Disparity Map")
axs[1, 0].axis("off")
fig.colorbar(im_disp, ax=axs[1, 0], fraction=0.046, pad=0.04)

im_depth = axs[1, 1].imshow(depth, cmap='inferno')
axs[1, 1].set_title("Depth Map")
axs[1, 1].axis("off")
fig.colorbar(im_depth, ax=axs[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()