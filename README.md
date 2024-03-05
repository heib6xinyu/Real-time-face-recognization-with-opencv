# Real-Time Face Recognition System

## Overview
This project develops a real-time face recognition system using OpenCV, demonstrating the application of computer vision techniques and machine learning algorithms for biometric identification. By integrating various Python scripts, the system prepares data, trains a facial recognition model, and then applies this model to recognize faces in real-time video feeds.

## Features
- **Real-time face detection and recognition**: Leverages OpenCV to identify faces in video streams instantly.
- **Custom training pipeline**: Includes scripts for preprocessing data, training a face recognition model, and predicting identities with the trained model.
- **Data preprocessing**: Automates the preparation of training data, ensuring the model learns from clean and relevant facial features.
- **Model training and serialization**: Facilitates the creation and storage of a facial recognition model, making it reusable across sessions.


## Technologies & Tools
- **Python 3.8.18**: The programming language used for scripting the entire pipeline. 
- **OpenCV 4.5.2.54**: Used for all image processing and face detection tasks, ensuring compatibility with the project's requirements.
- **Machine Learning**: Utilizing OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer for the recognition task.
- **File I/O**: For handling datasets and model files efficiently.


## Getting Started
### Prerequisites
- Python 3.8
- OpenCV library 4.5.2
- Numpy (optional, for array manipulation during preprocessing)

### Installation
1. Clone this repository to your local machine.
2. Install the required Python packages:
   ```bash
   pip install opencv-python numpy
   ```

### How to Use
I've updated the README to include the data preparation instructions as you've outlined. Here is the revised section:


## Data Preparation
To prepare your data for the face recognition model:

1. **Organize your face data**: Place your images in the `./data/raw_data/person_name/` directory, with the format `person_face.jpg`. Replace `person_name` with the actual name of the person to whom the face belongs. This naming convention helps in automating the data preprocessing step.

2. **Run `Preparing_the_model.py`**: This script prepares your data, making it ready for the training process.

3. **Execute `train_model.py`**: This will train the face recognition model with your preprocessed data.

4. **Finally, run `predict.py`**: To test the predictions of your model against new face images.

Ensure your dataset is well-organized and named according to the specified format for seamless training and prediction processes.



## Project Structure
- `preprocess.py`: Script for preprocessing training data.
- `train_model.py`: Contains code for training the face recognition model.
- `face_recognization_and_label.py`: Main script for real-time face detection and recognition.
- `predict.py`: Utility for making predictions with the trained model.
- `Preparing_the_model.py`: Helper script for setting up the recognition model.


