import cv2

img = cv2.imread('sane.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints
img_kp = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT Keypoints', img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()











# START
#   │
#   ├──► Load color image using OpenCV
#   │
#   ├──► Convert image to grayscale
#   │
#   ├──► Create a SIFT feature detector object
#   │
#   ├──► Detect keypoints and compute descriptors
#   │
#   ├──► Draw keypoints on the image
#   │
#   ├──► Display image with drawn keypoints
#   │
#   └──► Wait and exit
#  END
