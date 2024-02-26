import cv2
import os


#define file path
input_dir = "data\\raw_data"
output_dir = "data\\cropped_faces"


def read_images_from_folder(input_dir, output_dir, desired_size  = (256,256)):
    #create the output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Output directory created.")
    else:
        print("Output directory already exists.")
    
    #load the Haar Cascade face detection model
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    #process folders name after person id
    for subdir, dirs,files in os.walk(input_dir):
        person_id = os.path.basename(subdir) #extract the person's name from raw_data's subdir
        if subdir == input_dir:  # Skip the root directory
            continue
        output_person_folder = os.path.join(output_dir, person_id)
        if not os.path.exists(output_person_folder):
            os.makedirs(output_person_folder)  # Create a directory for each person
            print("Person output directory created.")
        else:
            print("Person output directory already exists.")
        for pic in files:
            if pic.lower().endswith(('.png','.jpg','.jpeg')): #check if the files are of acceptable format
                file_path = os.path.join(subdir,pic)
                print(f"Attempting to read: {file_path}")
                image = cv2.imread(file_path) #read the image
                if image is None:
                    print(f"Failed to read image at: {file_path}")
                    continue
                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.1,4)

                if len(faces) == 0:  # No faces detected
                    print(f"Failed to read face in picture: {file_path}")
                    continue

                #crop faces found
                # Sort faces based on the area (w * h) in descending order and select the largest one
                largest_face = max(faces, key=lambda x: x[2] * x[3])

                # Process only the largest face
                x, y, w, h = largest_face
                face = image[y:y+h, x:x+w]
                
                    
                #convert all images to jpeg
                output_image_path = os.path.join(output_person_folder, f"{person_id}_{pic}")
                output_image_path = output_image_path.replace(os.path.splitext(output_image_path)[1], ".jpg")  # Change extension to .jpg

                #save the image to the defined path
                cv2.imwrite(output_image_path, face)

#process_images_from_folder(input_dir, output_dir)