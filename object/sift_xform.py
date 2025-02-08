import cv2
import matplotlib.pyplot as plt

sift = cv2.SIFT_create()
path = '/home/roman/Downloads/fpv_datasets/outdoor_forward_1_davis_with_gt/img/'

file1 = 'image_0_1613.png'
img = cv2.imread(path + file1, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found. Please check the file path.")


keypoints, descriptors = sift.detectAndCompute(img, None)

print(f"Detected {len(keypoints)} keypoints")
if descriptors is not None:
    print(f"Descriptor shape: {descriptors.shape}")

print(descriptors)


img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints")
plt.axis("off")
plt.show()