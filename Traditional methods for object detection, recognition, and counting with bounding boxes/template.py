
import cv2

img = cv2.imread('group selfie.jpg')
template = cv2.imread('selfie.png')
h, w = template.shape[:2]

# Match template
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(res)

# Draw rectangle
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

cv2.imshow('Template Matching', img)
cv2.waitKey(0)
cv2.destroyAllWindows()











# Template Matching is a technique in computer vision where:

# A small image (template) is searched within a larger image (scene).

# It slides the template across the scene image and compares similarity at each location.

# The position with the best match is returned as the detection point.