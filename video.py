import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_fontalface_default.xml')


webcam = cv2.VideoCapture(0)
cv2.namedWindow("Face Detection")

# Iterate forever over the frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangle
    for face in face_coordinates:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

        cv2.imshow("Face Detection", frame)

    # This breaks on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
