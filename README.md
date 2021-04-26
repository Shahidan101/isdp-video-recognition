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
Test
## Usage
Test
## Contributing and Bug Reporting
