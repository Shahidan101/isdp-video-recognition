import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt 
import time
 
########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.50 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
#####################################

#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)

net = cv2.dnn.readNet("frozen_east_text_detection.pb")

while True:
    success, imgOriginal = cap.read()
    
    blob = cv2.dnn.blobFromImage(imgOriginal, 1.0, (width, height), (123.68, 116.78, 103.94), True, False)

    outputLayers = []

    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")
    net.setInput(blob)
    
    output = net.forward(outputLayers)
    scores = output[0]
    geometry = output[1]

    [boxes, confidences] = decode(scores, geometry, confThreshold)

    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)

    print("End of a loop")