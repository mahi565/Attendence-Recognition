# capture_samples.py

import cv2
import os

def capture_samples(output_dir):
    cam = cv2.VideoCapture(0)  # Open default camera (usually webcam)

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Capture Samples - Press Space to Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Space key pressed
            img_name = os.path.join(output_dir, f"sample_{len(os.listdir(output_dir)) + 1}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Saved: {img_name}")

        elif key == ord('q'):  # 'q' key pressed
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_directory = "dataset"
    capture_samples(output_directory)
