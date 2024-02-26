import cv2
import json
import sys
#load the trained model and face cascade

model_path = "model.yml"
face_cascade_path = "haarcascade_frontalface_default.xml"

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_path)

face_cascade = cv2.CascadeClassifier(face_cascade_path)

#Initialize the webcam
cap = cv2.VideoCapture(0)

frame_skip = 20
counter = 0
json_file_path = 'data\\label_dict.json'  # Update this path
with open(json_file_path, 'r') as file:
    name_to_label = json.load(file)

label_to_name = {v: k for k, v in name_to_label.items()}

try:
    while True:
        #capture frame by frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        counter += 1
    
        if counter < frame_skip:
            continue

        counter = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for(x, y, w, h) in faces:
            #Draw rectangle around the face
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)

            #crop the face and predict using the model
            roi_gray = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(roi_gray)
            person_name = label_to_name.get(label,"UNKNOWN")
            print(person_name)
            print(f"Predicted label: {person_name} with confidence: {confidence}")

            #display the label and confidence on the frame
            cv2.putText(frame, f"Label: {person_name}, {confidence: .2f}", (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),2)
    
        #display the resulting frame
        cv2.imshow("Frame",frame)

        #break the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

