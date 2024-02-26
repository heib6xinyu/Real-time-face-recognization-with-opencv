import cv2
import os

#recognizer = cv2.face.LBPHFaceRecognizer_create()
#dataset_path = "data\\pre_processed"

#initialize lists for storing faces samples and labels



def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    label_dict = {}
    dirs = os.listdir(data_folder_path)
    label = 0
    for dir_name in dirs:
        #if not dir_name.startswith("."):#igore files starting with .
        subdir_path = os.path.join(data_folder_path, dir_name)
        subdir_images_names = os.listdir(subdir_path)

        #assign a label to the subdir, which is person name, if not already assigned
        if dir_name not in label_dict:
            label_dict[dir_name] = label
            label += 1
        for image_name in subdir_images_names:
            #if image_name.startswith("."):
            #   continue
            image_path = os.path.join(subdir_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image at: {image_path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            data_label = label_dict[dir_name]
            faces.append(gray)
            labels.append(data_label)

    return faces, labels,label_dict