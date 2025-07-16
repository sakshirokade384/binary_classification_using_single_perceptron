import cv2
import numpy as np
import random

# === CONFIGURATION ===
# NOTE: Swapped left/right to fix stitch direction
left_image_path = r"C:\Users\suraj\OneDrive\Desktop\SAKSHI DOCUMENTS\right.png"
right_image_path = r"C:\Users\suraj\OneDrive\Desktop\SAKSHI DOCUMENTS\left.png"
SHOW_KEYPOINT_MATCHES = True  # Set to False to skip debug match display

# === LOAD IMAGES ===
img1 = cv2.imread(left_image_path)
img2 = cv2.imread(right_image_path)

if img1 is None or img2 is None:
    print("❌ Failed to load one or both images.")
    exit()

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# === DETECT KEYPOINTS ===
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"✅ Keypoints: left={len(kp1)}, right={len(kp2)}")

if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
    print("❌ No descriptors found in one or both images.")
    exit()

# === MATCH KEYPOINTS ===
def match_keypoints(kp1, kp2, des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good_raw = []
    good_pts = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_raw.append(m)
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            good_pts.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    return good_raw, good_pts

good_raw, good_pts = match_keypoints(kp1, kp2, des1, des2)
print(f"✅ Good matches found: {len(good_pts)}")

if len(good_pts) < 10:
    print("❌ Not enough matches for reliable stitching.")
    exit()

# === OPTIONAL: DISPLAY & SAVE MATCHES ===
if SHOW_KEYPOINT_MATCHES:
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_raw, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Keypoints and Matches", img_matches)
    cv2.imwrite("keypoints_and_matches.jpg", img_matches)
    cv2.waitKey(0)
    cv2.destroyWindow("Keypoints and Matches")

# === HOMOGRAPHY ===
def compute_homography(points):
    A = []
    for pt in points:
        x, y, X, Y = pt
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    return H / H[2, 2]

# === RANSAC ===
def ransac(points, max_iter=5000, threshold=5.0):
    best_inliers = []
    best_H = None
    if len(points) < 4:
        return None
    for _ in range(max_iter):
        try:
            sample = random.sample(points, 4)
            H = compute_homography(sample)
            if not np.all(np.isfinite(H)):
                continue
        except:
            continue
        inliers = []
        for pt in points:
            p1 = np.array([pt[0], pt[1], 1.0]).reshape(3, 1)
            p2 = np.array([pt[2], pt[3], 1.0]).reshape(3, 1)
            Hp = H @ p1
            if Hp[2] == 0:
                continue
            Hp = Hp / Hp[2]
            dist = np.linalg.norm(Hp - p2)
            if dist < threshold:
                inliers.append(pt)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
    return (best_inliers, best_H) if best_H is not None else None

# === ESTIMATE FINAL HOMOGRAPHY ===
ransac_result = ransac(good_pts)
if ransac_result is None:
    print("❌ RANSAC failed. Could not compute homography.")
    exit()

inliers, H = ransac_result
print(f"✅ RANSAC inliers: {len(inliers)}")

if len(inliers) < 4:
    print("❌ Not enough inliers for final homography.")
    exit()

# === REFINE HOMOGRAPHY ===
H = compute_homography(inliers)
print("✅ Refined homography matrix computed.")

# === WARPING AND STITCHING ===
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
warped_corners = cv2.perspectiveTransform(corners_img2, H)
all_corners = np.concatenate((warped_corners,
                              np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)),
                             axis=0)

[xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

translate = [-xmin, -ymin]
T = np.array([[1, 0, translate[0]],
              [0, 1, translate[1]],
              [0, 0, 1]])

result_size = (xmax - xmin, ymax - ymin)
result = np.zeros((result_size[1], result_size[0], 3), dtype=np.uint8)

cv2.warpPerspective(img2, T @ H, result_size, dst=result, borderMode=cv2.BORDER_TRANSPARENT)

# === OVERLAY FIRST IMAGE ===
start_y, end_y = translate[1], translate[1] + h1
start_x, end_x = translate[0], translate[0] + w1
result[start_y:end_y, start_x:end_x] = img1

# === CROP BLACK BORDERS ===
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
coords = cv2.findNonZero(thresh)
if coords is not None and len(coords) > 0:
    x, y, w, h = cv2.boundingRect(coords)
    cropped_result = result[y:y+h, x:x+w]
else:
    cropped_result = result
    print("⚠️ No cropping applied (blank canvas?)")

# === SAVE AND DISPLAY ===
cv2.imwrite("stitched_panorama.jpg", cropped_result)
print("✅ Stitched panorama saved as stitched_panorama.jpg")

cv2.imshow("Stitched Panorama", cropped_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
