import numpy as np
import cv2
import pickle
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
 
checker = []
guess = 50

circle_flag = 0
circleList = []

#### LOAD THE TRAINNED MODEL
model = tf.keras.models.load_model('model_trained.h5')
# model = tf.keras.models.load_model('isdp_number_recognition.h5')
 
#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    outcome = 0

    probVal = 0

    success, imgOriginal = cap.read()

    # Converts frame to RGB. More suitable for showing using matplotlib. Assigns to a variable
    img_circle = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)

    # Now we need to convert to grayscale, then blur the image
    img_circle = cv2.cvtColor(img_circle, cv2.COLOR_BGR2GRAY)

    img_circle = cv2.medianBlur(img_circle, 5)

    # Find circles in the grayscale frame using Hough Transform
    circles = cv2.HoughCircles(image=img_circle, method=cv2.HOUGH_GRADIENT, dp=0.9, minDist=600, param1=110, param2=39, minRadius=100 ,maxRadius=400)

    if circles is not None:
        # Now draw and show the circles that were detected
        for co, i in enumerate(circles[0, :], start=1):

            x = i[0]
            y = i[1]
            r = int(i[2])

            # Draw outer circle in green
            cv2.circle(imgOriginal, (i[0], i[1]), int(i[2]), (0, 255, 0), 2)

            rectX_low = int(i[0]) - int(i[2])
            rectY_low = int(i[1]) - int(i[2])

            rectX_high = int(i[0]) + 1 - int(i[2])
            rectY_high = int(i[1]) + 1 - int(i[2])

            crop_img = imgOriginal[rectY_low:(rectY_high + 2 * r), rectX_low:(rectX_high + 2 * r)]

            # Draw center of circle in red
            cv2.circle(imgOriginal, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Print number of circles detected
        # print("Number of circles detected:", co)

        # Add circle count into list, then do averaging to find most likely number
        circleList.append(co)

        crop_height, crop_width, something = crop_img.shape

        if (crop_height > 100) and (crop_width > 100):
            img = np.asarray(crop_img)
            # print(img.shape)
            img = cv2.resize(img,(32,32))
            img = preProcessing(img)
            # cv2.imshow("Processsed Image",img)
            img = img.reshape(1,32,32,1)
            #### PREDICT
            classIndex = int(model.predict_classes(img))
            #print(classIndex)
            predictions = model.predict(img)
            #print(predictions)
            probVal= np.amax(predictions)
            # print(classIndex,probVal)

    # Print out outcome
    if len(circleList) > 20:
        shape = "Sphere"
        circle_flag = 1
        circleList.clear()

    if probVal> threshold:
        if classIndex == guess:
            checker.append(classIndex)
        else:
            guess = classIndex

        if len(checker) > 15 and circle_flag == 1:
            largest = checker.count(list(set(checker))[0])
            largestitem = list(set(checker))[0]

            for i in range(1, len(list(set(checker)))):
                if checker.count(list(set(checker))[i]) > largest:
                    largest = checker.count(list(set(checker))[i])
                    largestitem = list(set(checker))[i]

            classIndex = largestitem

            print("Mineral Type:", shape)
            print("Number on Mineral:", classIndex)
            print()
            circle_flag = 0
            checker.clear()

        text1 = "Predicted: " + str(classIndex)
        text2 = "Probability: " + str(probVal)
        cv2.putText(imgOriginal,text1,
                    (50,430),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text2,
                    (50,460),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
    else:
        text1 = "Predicted: NONE"
        text2 = "Probability: NONE"
        cv2.putText(imgOriginal,text1,
                    (50,430),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text2,
                    (50,460),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)

    cv2.imshow("Original Image",imgOriginal)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break