import cv2

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Use full path or relative path
img_path = r"C:\Users\suraj\data_visualisation\imgs\selfie.jpeg"  # <-- Change this to your actual image path
img = cv2.imread(img_path)

if img is None:
    print("Error: Image not found or path is incorrect")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Haar Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








# Start
#   │
#   ▼
# Import OpenCV
#   │
#   ▼
# Load Haar Cascade Classifier (face)
#   │
#   ▼
# Load Image from Path
#   │
#   ├──► If image is None:
#   │        └──► Print Error and Exit
#   │
#   ▼
# Convert Image to Grayscale
#   │
#   ▼
# Detect Faces using detectMultiScale()
#   │
#   ▼
# For each detected face:
#   └──► Draw Rectangle on Face
#   │
#   ▼
# Display Image with Detected Faces
#   │
#   ▼
# Wait for Key Press
#   │
#   ▼
# Close All Windows
#   │
#   ▼
# End
