import cv2
import os



def resize_image(image,size = (128,128)):
    return cv2.resize(image,size)

def normalize_image(image):
    return image/225.0

def rotate_image(image,angle=90):
    (h,w) =image.shape[:2]
    center =(w//2,h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M,(w,h))
    return rotated

def flip_image(image):
    return cv2.flip(image,1) #1 for horizontal flipping

def process_and_augment_image(file_path, output_person_dir, resize_to = (128,128), normalize = True):
    #resize
    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to load image: {file_path}")
        return
    image = resize_image(image,size = resize_to)
    #normalize
    if normalize:
        image = normalize_image(image)

    base_name = os.path.basename(file_path)
    cv2.imwrite(os.path.join(output_person_dir, base_name), image * 255) #convert back

    #augmentations:
    #rotate
    rotated_image = rotate_image(image, angle = 90)
    cv2.imwrite(os.path.join(output_person_dir, f"rotated_{base_name}"), rotated_image * 255)

    #flip
    flipped_image = flip_image(image)
    cv2.imwrite(os.path.join(output_person_dir, f"flipped_{base_name}"), flipped_image * 255)

def process_images_from_folder(input_dir,output_dir,resize_to = (128,128)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Preprocess output directory created.")
    else:
        print("Preprocess output directory already exists.")

    for subdir, dirs, files in os.walk(input_dir):
        person_id = os.path.basename(subdir)
        if subdir == input_dir: #skip the root input directory
            continue
        output_person_dir = os.path.join(output_dir, person_id)
        if not os.path.exists(output_person_dir):
            os.makedirs(output_person_dir)

        for pic in files:
            if pic.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, pic)
                process_and_augment_image(file_path, output_person_dir, resize_to=resize_to)

