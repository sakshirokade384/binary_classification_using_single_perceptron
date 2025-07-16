import cv2
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# # Use correct and full path
# img_path = r"C:\Users\suraj\data_visualisation\imgs\selfie.jpeg"

# Load the image in grayscale
img = cv2.imread('texture.jpg', cv2.IMREAD_GRAYSCALE)

# Check if image loaded correctly
if img is None:
    print("Error: Could not load image. Check the file path.")
else:
    # Apply Local Binary Pattern
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")

    # Display the result
    plt.imshow(lbp, cmap='gray')
    plt.title('LBP Image')
    plt.axis('off')
    plt.show()












# What LBP Is Good At
# Capability	How LBP Helps
# Texture classification	Distinguishes patterns in surfaces/fabrics/walls
# Face recognition	Captures skin texture patterns (e.g. wrinkles, pores)
# Feature extraction	Converts image into a histogram of texture codes
# Fast + simple	Works in real-time on CPU (no deep learning needed)

