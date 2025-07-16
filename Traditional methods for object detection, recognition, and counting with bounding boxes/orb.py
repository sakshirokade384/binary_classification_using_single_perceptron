import cv2

img = cv2.imread('hoome.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)

img_orb = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

cv2.imshow('ORB Keypoints', img_orb)
cv2.waitKey(0)
cv2.destroyAllWindows()



















# START
#   │
#   ├─► Read image
#   ├─► Convert to grayscale
#   ├─► Create ORB detector
#   ├─► Detect keypoints and descriptors
#   ├─► Draw keypoints
#   ├─► Display the result
#   └─► Wait for key and exit
#  END
