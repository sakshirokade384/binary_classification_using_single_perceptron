import cv2

# Initialize HOG descriptor and SVM
import cv2

# Initialize HOG descriptor and set the default people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())



img_path  = r"C:\Users\suraj\data_visualisation\imgs\street.png"
img = cv2.imread(img_path)

# Detect people
boxes, _ = hog.detectMultiScale(img, winStride=(3,3))

# Draw rectangles
for (x, y, w, h) in boxes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('HOG + SVM Person Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()













# START
#   │
#   ├──► Import OpenCV
#   │
#   ├──► Initialize HOGDescriptor
#   │        └── Set SVM Detector for people
#   │
#   ├──► Load image from disk
#   │        └── If image not found → Print error and EXIT
#   │
#   ├──► (Optional) Resize or convert image to grayscale
#   │
#   ├──► Detect people using hog.detectMultiScale()
#   │        └── Sliding window + multi-scale detection
#   │
#   ├──► For each detected bounding box:
#   │        └── Draw rectangle on the image
#   │
#   ├──► Display result image in window
#   │
#   ├──► Wait for key press
#   │
#   └──► Close all OpenCV windows
#   │
#  END
