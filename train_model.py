# train_model.py

import os
import numpy as np
import face_recognition

def train_model(dataset_dir, model_save_path):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_encodings.append(encoding)
                known_names.append(person_name)

    # Save encodings and names
    np.save(model_save_path + "_encodings.npy", known_encodings)
    np.save(model_save_path + "_names.npy", known_names)
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    dataset_directory = "dataset"
    model_save_path = "models/face_recognition_model"
    train_model(dataset_directory, model_save_path)
