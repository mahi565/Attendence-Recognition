# recognize_attendance.py

import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os

def load_known_faces(model_dir):
    encodings = np.load(os.path.join(model_dir, "face_recognition_model_encodings.npy"))
    names = np.load(os.path.join(model_dir, "face_recognition_model_names.npy"))
    return encodings, names

def mark_attendance(name):
    with open("attendance.log", "a") as f:
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{time_now}\n")

def recognize_faces(model_dir):
    encodings, names = load_known_faces(model_dir)
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]

            # Draw rectangle and label on the face
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Mark attendance if recognized
            if name != "Unknown":
                mark_attendance(name)

        cv2.imshow("Recognizing Attendance - Press 'q' to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_directory = "models"
    recognize_faces(model_directory)
