# Form (Sphere & Cylinder) and Digit Recognition using OpenCV and Tensorflow
This repository contains scripts that enable Form and Digit Recognition. The software uses OpenCV functionality and deep learning on Tensorflow to operate. Originally, the scripts are developed for the purpose of fulfilling the course requirements of Integrated System Design Project (ISDP). The scripts are intended to be run on an autonomous robot powered by a Raspberry Pi 4 (4.00 GB RAM).

Part of the robot's mission (namely Task 3) involves receiving a sequence of numbers provided by the operator, then navigating the obstacle course to locate the objects labeled with the provided numbers. The robot needs to collect and transport the objects (acting as minerals) according to the sequence provided. Aside from transporting the minerals following the sequence, the robot also needs to recognise the form of the minerals (whether the mineral is a sphere or cylinder), then transmit the data back to the base station (a mobile application).

## Revision History
As of 26/04/2021:
- Detection of circles (and thus, spheres) in a live camera feed using Hough Transform
- Digit (0 to 9) recognition using Keras & Tensorflow
- Added output counters and mean calculation (of lists of output) to increase chances of yielding correct outputs
- Merging circle detection and digit recognition. Using circle detection to crop part of the frame containing the digit, then feeding the cropped image into the Tensorflow model. The morphological processing involved ensures clean test data is fed to the classification model, increasing the accuracy of prediction
 
## Installation
### Python Libraries
- Keras 2.4.3
- matplotlib 3.3.3
- numpy 1.19.5
- opencv-python 4.5.1.48
- scikit-learn 0.24.1
- tensorflow 2.4.1

The libraries above are essential in allowing the software to work. While other versions of the libraries may allow the software to work correctly, the versions selected are what worked for me. The libraries above can be installed through a [pip installation](https://pip.pypa.io/en/stable/) on your preferred terminal. Example is as below:

```
pip install tensorflow==2.4.1
```

### Dataset
The dataset is compressed into **myData.rar**. Extract the RAR file to obtain the full dataset used to train the model. The folder and file hierarchy is shown below. Ensure that the extracted dataset folder (myData) is placed in the same directory as the remaining scripts for the software.

```
myData
--> 0
--> 1
--> 2
--> 3
--> 4
--> 5
--> 6
--> 7
--> 8
--> 9
```

### Training the model (numberRecognition_train)
The script loads the image dataset from the **myData** folder. The dataset is split into training, test, and validation sets, with a test and validation ratio of 20%. Before using the dataset, the images are preprocessed by converting to Grayscale and implementing Histogram Equalisation. Additional preprocessing and image augmentation is done using the Keras image preprocessor. The labels are one-hot encoded.

The summary of the Convolutional Neural Network is provided below.

```
---------------------------------------------------
Layer (type)                    Output Shape
===================================================
conv2d (Conv2D)                 (None, 28, 28, 60)
conv2d_1 (Conv2D)               (None, 24, 24, 60)
max_pooling2d (MaxPooling2D)    (None, 12, 12, 60)
conv2d_2 (Conv2D)               (None, 10, 10, 30)
conv2d_3 (Conv2D)               (None, 8, 8, 30)
max_pooling2d_1 (MaxPooling2D)  (None, 4, 4, 30)
dropout (Dropout)               (None, 4, 4, 30)
flatten (Flatten)               (None, 480)
dense (Dense)                   (None, 500)
dropout_1 (Dropout)             (None, 500)
dense_1 (Dense)                 (None, 10)
```

After training the model using the dataset provided, the model is output into a h5 file named **model_trained.h5**.

### Video Processing, Circle Detection and Digit Recognition (numberRecognitionWithCircle.py)
The script starts with selecting the camera and beginning the video capture. The trained model from the previous script is loaded from the h5 file. Each frame of the live camera feed is processed as an image. The frame is first converted to RGB, which is more suitable for showing using matplotlib. The image is then Grayscaled and Median Blurred. 

Using Hough Transform, the circles in the frame are detected with specified minimum distance between circles, minimum radius, and maximum radius. The circles are stored into a list. When the list is not empty (meaning circles are present in the frame), the circles are drawn in the frame in green with the center point in red. Using the coordinates and boundaries of the circle, the rectangular area of the detected circles are cropped into an image variable. The cropped image is fed into the loaded model. The model predicts the digit in the cropped image and produces a probability value for the prediction. 

In order to improve the output of the prediction, certain measures are implemented to check for the object to be in view of the camera for a certain duration. This ensures that a sudden or slight change in ambient light or orientation does not cause an instant prediction of wrong values. The measures implemented make use of appending lists of the detected circles and predicted values, checkng the sizes of the lists, as well as yielding the most frequently appeared value in the lists to identify the correct outputs.

## Usage
### The fastest way to go about it
Clone the following files. Ensure that the files are in the same directory.
- model_trained.h5
- numberRecognitionWithCircle.py

To run the software, run **numberRecognitionWithCircle.py**. This can be done in the terminal as shown.
```
python numberRecognitionWithCircle.py
```

### The slower way to go about it
Clone the following files/folders. Ensure that the files are in the same directory.
- Dataset
- numberRecognition_train.py
- numberRecognitionWithCircle.py
Prepare the dataset as highlighted in *Installation --> Dataset*. Run **numberRecognition_train.py* to train the model with the image dataset. Once the h5 file is produced, run **numberRecognitionWithCircle.py**.

## Contributing and Bug Reporting
For contributions, development, and issues, please contact [shahidan.utp@gmail.com](mailto:shahidan.utp@gmail.com).
