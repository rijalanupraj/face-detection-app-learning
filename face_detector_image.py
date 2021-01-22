import cv2

# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Load some pre-trained data on face frontals from openCV (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_fontalface_default.xml')

# Reading an image to detect faces
img = cv2.imread('anup.JPG')

# Convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangle
for face in face_coordinates:
    (x, y, w, h) = face
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

#
cv2.imshow('Face Detector', img)


cv2.waitKey()
