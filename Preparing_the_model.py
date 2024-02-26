import cv2
from face_recognization_and_label import *
from preprocess import *
from train_model import *
import numpy as np
import json

#define file path
input_dir = "data\\raw_data"
output_dir = "data\\cropped_faces"
output_processed_dir = "data\\pre_processed"

#read_images_from_folder(input_dir, output_dir)
process_images_from_folder(output_dir, output_processed_dir)

faces, labels, label_dict= prepare_training_data(output_processed_dir)

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
#train the recognizer
recognizer.train(faces, labels)
recognizer.write('model.yml')

with open("data\\label_dict.json", "w") as file:
    json.dump(label_dict, file)
