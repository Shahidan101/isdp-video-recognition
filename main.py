import numpy as np
import imutils
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt 
import time
import sys
import math
import os
from os import listdir
from os.path import isfile, join

#### LOAD THE TRAINNED MODEL
model = tf.keras.models.load_model('model_trained.h5')

def FlagToShape(number):
    if number == 0:
        outShape = "NONE"
    elif number == 1:
        outShape = "SPHERE"
    elif number == 2:
        outShape = "CYLINDER"
    return outShape

def distanceBetweenPoints(point1, point2):
    distance = math.sqrt(pow((point2[0] - point1[0]), 2) + pow((point2[1] - point1[1]), 2))
    return distance
 
# [========== THIS PREPROCESSING FUNCTION IS OBSOLETE NOW ==========]
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
# [========== THIS PREPROCESSING FUNCTION IS OBSOLETE NOW ==========]

#### Just a sharpening function
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

#### PREPORCESSING FUNCTION
def newpreProcessing(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,0])
    upper = np.array([179,174,176])

    # FOR THE DIY TEST ITEMS
    # lower = np.array([90,29,0])
    # upper = np.array([138,255,255])

    # lower = np.array([81,36,0])
    # upper = np.array([165,255,255])

    mask = cv2.inRange(image, lower, upper)
    mask = 255 - mask
    cv2.imwrite("eBB2Q.jpg", mask)
    cv2.imwrite("luraB.jpg", image)
    mask = cv2.imread('eBB2Q.jpg')
    image = cv2.imread('luraB.jpg')
    cv2.imwrite("eBB2Q.jpg", mask)
    cv2.imwrite("luraB.jpg", image)
    mask = cv2.imread('eBB2Q.jpg')
    image = cv2.imread('luraB.jpg')
    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    combinedImage = np.copy(image)
    combinedImage[(mask==255).all(-1)] = [255,255,255]
    combinedImage = cv2.cvtColor(combinedImage, cv2.COLOR_BGR2GRAY)
    img = combinedImage
    img = cv2.equalizeHist(img)
    img = img/255
    return img    

#### COLOUR FILTERING FUNCTIONS
def redFilter(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([126,67,0])
    upper = np.array([179,255,255])
    mask = cv2.inRange(image, lower, upper)
    contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # not copying here will throw an error
    if len(contours) != 0:
        largest_area = cv2.contourArea(contours[0])
        largest_cnt = contours[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_area = area
                largest_cnt = cnt
        rect = cv2.boundingRect(largest_cnt)
        x, y, w, h = rect
        centre = (x + w/2, y + h/2)
        widthHeight = w, h

        # To determine cylinder
        # if h > w:

        rect2 = cv2.rectangle(img.copy(),(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("Bounded Red Mask", rect2)
    mask = 255 - mask
    cv2.imwrite("eBB2Q.jpg", mask)
    cv2.imwrite("luraB.jpg", image)
    mask = cv2.imread('eBB2Q.jpg')
    image = cv2.imread('luraB.jpg')
    cv2.imwrite("eBB2Q.jpg", mask)
    cv2.imwrite("luraB.jpg", image)
    mask = cv2.imread('eBB2Q.jpg')
    image = cv2.imread('luraB.jpg')
    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    combinedImage = np.copy(image)
    combinedImage[(mask==255).all(-1)] = [0,0,0]
    combinedImage = cv2.cvtColor(combinedImage, cv2.COLOR_BGR2GRAY)
    img = combinedImage
    img = cv2.equalizeHist(img)
    img = img/255

    # If area of mask is large, this is the object in front of you
    # Compare areas between filter functions
    if len(contours) != 0:
        return largest_area, centre, widthHeight, rect
    else:
        return 0, (0,0), (0,0), (0,0,0,0)         # ISSUE: When there are no colours at all!

def greenFilter(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([32,146,0])
    upper = np.array([76,255,255])
    # lower = np.array([50,146,0])
    # upper = np.array([76,255,255])
    mask = cv2.inRange(image, lower, upper)
    cv2.imshow("Green Mask", mask)
    contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # not copying here will throw an error
    if len(contours) != 0:
        largest_area = cv2.contourArea(contours[0])
        largest_cnt = contours[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_area = area
                largest_cnt = cnt
        rect = cv2.boundingRect(largest_cnt)
        x, y, w, h = rect
        centre = (x + w/2, y + h/2)
        widthHeight = w, h
        rect2 = cv2.rectangle(img.copy(),(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("Bounded Green Mask", rect2)
    mask = 255 - mask
    cv2.imwrite("eBB2Q.jpg", mask)
    cv2.imwrite("luraB.jpg", image)
    mask = cv2.imread('eBB2Q.jpg')
    image = cv2.imread('luraB.jpg')
    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    combinedImage = np.copy(image)
    combinedImage[(mask==255).all(-1)] = [0,0,0]
    combinedImage = cv2.cvtColor(combinedImage, cv2.COLOR_BGR2GRAY)
    img = combinedImage
    img = cv2.equalizeHist(img)
    img = img/255

    # If area of mask is large, this is the object in front of you
    # Compare areas between filter functions
    if len(contours) != 0:
        return largest_area, centre, widthHeight, rect
    else:
        return 0, (0,0), (0,0), (0,0,0,0)         # ISSUE: When there are no colours at all!

def blueFilter(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([90,154,0])
    upper = np.array([175,255,255])
    mask = cv2.inRange(image, lower, upper)
    contours,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # not copying here will throw an error
    if len(contours) != 0:
        largest_area = cv2.contourArea(contours[0])
        largest_cnt = contours[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_area = area
                largest_cnt = cnt
        rect = cv2.boundingRect(largest_cnt)
        x, y, w, h = rect
        centre = (x + w/2, y + h/2)
        widthHeight = w, h

        # To determine cylinder
        # if h > w:

        rect2 = cv2.rectangle(img.copy(),(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("Bounded Blue Mask", rect2)
    mask = 255 - mask
    cv2.imwrite("eBB2Q.jpg", mask)
    cv2.imwrite("luraB.jpg", image)
    mask = cv2.imread('eBB2Q.jpg')
    image = cv2.imread('luraB.jpg')
    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    combinedImage = np.copy(image)
    combinedImage[(mask==255).all(-1)] = [0,0,0]
    combinedImage = cv2.cvtColor(combinedImage, cv2.COLOR_BGR2GRAY)
    img = combinedImage
    img = cv2.equalizeHist(img)
    img = img/255

    # If area of mask is large, this is the object in front of you
    # Compare areas between filter functions
    if len(contours) != 0:
        return largest_area, centre, widthHeight, rect
    else:
        return 0, (0,0), (0,0), (0,0,0,0)         # ISSUE: When there are no colours at all!

# [========== WORK IN PROGRESS ==========]

#### EDGE PREPORCESSING FUNCTION
def EdgepreProcessing(img):
    # cv2.imshow("Before Processing", img)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Below is the combination for HSV values
    lower = np.array([0,0,0])
    upper = np.array([32,255,255])
    mask = cv2.inRange(image, lower, upper)
    mask = 255 - mask
    # cv2.imshow("The Mask", mask)
    cv2.imwrite("edgeMask.jpg", mask)
    cv2.imwrite("edgeRecoloured.jpg", image)
    mask = cv2.imread("edgeMask.jpg")
    face = cv2.imread("edgeRecoloured.jpg")
    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    # copy where we'll assign the new values
    green_hair = np.copy(face)
    # boolean indexing and assignment based on mask
    green_hair[(mask==255).all(-1)] = [255,255,255]
    green_hair = cv2.cvtColor(green_hair, cv2.COLOR_BGR2GRAY)
    img = green_hair
    img = cv2.equalizeHist(img)
    img = img/255
    return img 
# [========== WORK IN PROGRESS ==========]   


def imageProcessing(imageObject):

    ########### PARAMETERS ##############
    width = 640
    height = 480
    threshold = 0.50 # MINIMUM PROBABILITY TO CLASSIFY
    areaThreshold = 23000
    # cameraNo = 0

    # cap = cv2.VideoCapture(cameraNo)
    # cap.set(3,width)
    # cap.set(4,height)
    #####################################

    # --------------- PART OF TESTBENCH --------------- 

    imageSingle = imageObject[7:]
    testbenchList = []
    testbenchList.append("Image : " + imageSingle)

    testPassFail = []
    testPassFail.append("Image : " + imageSingle)
    imageSingle = imageSingle.replace(".jpg", "")
    fullTestComponents = []

    fullTestComponents = imageSingle.split('_')

    correctColour = fullTestComponents[0].upper()
    correctObject = "MINERAL"
    correctShape = fullTestComponents[1]
    if correctShape == "circle":
        correctShape = "SPHERE"
    elif correctShape == "cylinder":
        correctShape = correctShape.upper()
    correctNumber = int(fullTestComponents[2])

    # --------------- PART OF TESTBENCH ---------------
    
    checker = []
    circleList = []
    guess = 50
    circle_flag = 0
    cylinderProposal = 0            # In case we read Cylinder and Circle at the same time
    imageCounter = 0                # To do a loop of running all images
    imageContainer = []             # To count the number of images loaded, basically a duration counter

    outcome = 0
    shapeFlag = 0       # 0 for NONE, 1 for SPHERE, 2 for CYLINDER

    probValCircle = 0
    probValCylinder = 0

    # [========== SECTION FULL OF IMAGES TO TEST ==========]   

    # imageDirPath = os.path.join('images')
    # imageNames = [f for f in listdir(imageDirPath) if isfile(join(imageDirPath, f))]
    # imagePath = os.path.join('images', imageNames[imageCounter])
    
    #### CREATE CAMERA OBJECT
    # success, imgOriginal = cap.read()

    imgOriginal = cv2.imread(imageObject)
    # imageContainer.append(imageNames[imageCounter])

    # imgOriginal = cv2.imread('red_circle_4_2.jpg')
    # imgOriginal = cv2.imread('red_circle_1_1.jpg')
    # imgOriginal = cv2.imread('red_circle_1_2.jpg')
    # imgOriginal = cv2.imread('red_circle_1_7.jpg')
    # imgOriginal = cv2.imread('red_circle_1_14.jpg')
    # imgOriginal = cv2.imread('red_circle_1_19.jpg')
    # imgOriginal = cv2.imread('blue_circle_9_20.jpg')
    # imgOriginal = cv2.imread('blue_circle_9_21.jpg')
    # imgOriginal = cv2.imread('red_cylinder_3_1.jpg')
    # imgOriginal = cv2.imread('testWhite.jpg')

    # imgOriginal = unsharp_mask(imgOriginal)

    # if imgOriginal.shape != (480, 640, 3):
    #     oriWidth = imgOriginal.shape[0]
    #     oriHeight = imgOriginal.shape[1]
    #     aspectRatio = oriWidth / oriHeight
    #     newWidth = aspectRatio * 480
    #     imgOriginal = cv2.resize(imgOriginal, (480, int(newWidth)))

    # [========== SECTION FULL OF IMAGES TO TEST ==========]   

    areaRed, centreRed, widthHeightRed, rectRed = redFilter(imgOriginal.copy())
    areaGreen, centreGreen, widthHeightGreen, rectGreen = greenFilter(imgOriginal.copy())
    areaBlue, centreBlue, widthHeightBlue, rectBlue = blueFilter(imgOriginal.copy())

    # areaRed = int(areaRed)
    # areaGreen = int(areaGreen)
    # areaBlue = int(areaBlue)

    areaList = []
    areaRemoveList = []
    areaRemoveIndexList = []
    originalCentreList = []
    centreList = []
    centreListToPop = []
    widthHeightList = []
    originalWidthHeightList = []
    remainingCentreList = [0, 1, 2]
    remainingCentres = [0, 1, 2]
    areaListCounter = 0

    areaList.append(areaRed)
    areaList.append(areaGreen)
    areaList.append(areaBlue)

    centreList.append(centreRed)
    centreList.append(centreGreen)
    centreList.append(centreBlue)

    originalCentreList.append(centreRed)
    originalCentreList.append(centreGreen)
    originalCentreList.append(centreBlue)

    originalWidthHeightList.append(widthHeightRed)
    originalWidthHeightList.append(widthHeightGreen)
    originalWidthHeightList.append(widthHeightBlue)

    # ===== Filtering Colour Filter based on area ===== #
    for colourFilterAreas in areaList:
        if colourFilterAreas < areaThreshold:
            areaRemoveList.append(colourFilterAreas)
            areaRemoveIndexList.append(areaListCounter)
        areaListCounter = areaListCounter + 1

    for areas in areaRemoveList:
        areaList.remove(areas)

    for indexes in areaRemoveIndexList:
        remainingCentres.remove(indexes)
        centreListToPop.append(indexes)
    
    for stuff in centreListToPop:
        remainingCentreList.remove(stuff)

    centreList = []
    widthHeightList = []
    for numbers in remainingCentreList:
        centreList.append(originalCentreList[numbers])
        widthHeightList.append(originalWidthHeightList[numbers])
    # ===== Filtering Colour Filter based on area ===== #

    # ===== Assessing based on closer to centre, while considering area ===== #
    screenCentre = width / 2, height / 2

    # Initialise colour and object type variable
    colourInFront = "NONE"
    objectType = "NONE"
    classIndexCylinder = 99
    probValCylinder = 0
    classIndexCircle = 99
    probValCircle = 0

    if len(remainingCentres) != 0:
        shortestToCentre = distanceBetweenPoints(screenCentre, centreList[0])
        colourInFrontIndex = remainingCentres[0]

        for colourNumbers in remainingCentres:
            colourFilterCentres = originalCentreList[colourNumbers]
            distanceWithCentre = distanceBetweenPoints(screenCentre, colourFilterCentres)
            if distanceWithCentre < shortestToCentre:
                shortestToCentre = distanceWithCentre
                colourInFrontIndex = colourNumbers

        width, height = originalWidthHeightList[colourInFrontIndex]

        if colourInFrontIndex == 0:
            colourInFront = "RED"
        elif colourInFrontIndex == 1:
            colourInFront = "GREEN"
        elif colourInFrontIndex == 2:
            colourInFront = "BLUE"

        if colourInFront == "RED":
            objectType = "MINERAL"
        elif colourInFront == "BLUE":
            objectType = "MINERAL"
        elif colourInFront == "GREEN":
            objectType = "OBSTACLE"

        if width <= height * 0.7:
            # cylinderProposal = 1          # In case we read Cylinder and Circle at the same time
            if colourInFrontIndex == 0:
                predictionRect = rectRed
            elif colourInFrontIndex == 1:
                predictionRect = rectGreen
            elif colourInFrontIndex == 2:
                predictionRect = rectBlue

            x, y, w, h = predictionRect
            crop_img = imgOriginal.copy()[y:y+h, x:x+w]
            
            img2 = crop_img.copy()
            img2 = newpreProcessing(img2)

            x = originalCentreList[colourInFrontIndex][0] - (w / 2)
            y = originalCentreList[colourInFrontIndex][1] - (w / 2)
            h = w

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            img = imgOriginal.copy()[y:y+h, x:x+w]

            img = np.asarray(img)
            img = cv2.resize(img,(32,32))
            cv2.imshow("Resized Processsed Image Cylinder",img)
            img = newpreProcessing(img)
            cv2.imshow("Processsed Image Cylinder",img2)
            img = img.reshape(1,32,32,1)
            #### PREDICT
            classIndexCylinder = int(model.predict_classes(img))
            predictionsCylinder = model.predict(img)
            probValCylinder= np.amax(predictionsCylinder)
            shapeFlag = 2

            imgCylinder = imgOriginal.copy()

            if probValCylinder > threshold:
                # Show the predicted digit and probability value in the camera feed
                text1 = "Predicted: " + str(classIndexCylinder)
                text2 = "Probability: " + str(probValCylinder)
                text3 = "Shape: " + FlagToShape(shapeFlag)
                text4 = "Object Type: " + objectType
                text5 = "Colour: " + colourInFront
                # Display the text on the original frame (directly recorded from camera)
                cv2.putText(imgCylinder,text1,
                            (50,340),cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),1)
                cv2.putText(imgCylinder,text2,
                            (50,370),cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),1)
                cv2.putText(imgCylinder,text3,
                            (50,400),cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),1)
                cv2.putText(imgCylinder,text4,
                            (50,430),cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),1)
                cv2.putText(imgCylinder,text5,
                            (50,460),cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),1)
            else:
                # If the probability value does not exceed the threshold, display NONE for predicted value and probability value
                text1 = "Predicted: NONE"
                text2 = "Probability: NONE"
                text3 = "Shape: " + FlagToShape(shapeFlag)
                text4 = "Object Type: " + objectType
                text5 = "Colour: " + colourInFront
                cv2.putText(imgCylinder,text1,
                            (50,340),cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),1)
                cv2.putText(imgCylinder,text2,
                            (50,370),cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),1)
                cv2.putText(imgCylinder,text3,
                            (50,400),cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),1)
                cv2.putText(imgCylinder,text4,
                            (50,430),cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),1)
                cv2.putText(imgCylinder,text5,
                            (50,460),cv2.FONT_HERSHEY_DUPLEX,
                            1,(255,0,0),1)

            # Show the original frame
            cv2.imshow("Original Image With Cylinder",imgCylinder)

    # ===== Assessing based on closer to centre, while considering area ===== #

    # --------------- PART OF TESTBENCH ---------------
    # After Colour Detection and Cylinder Recognition Blocks
    testbenchList.append("------------------------------")
    testbenchList.append("AFTER COLOUR DETECTION AND CYLINDER RECOGNITION")
    testbenchList.append("------------------------------")
    testbenchList.append("Colour: " + colourInFront)
    testbenchList.append("Object Type: " + objectType)
    testbenchList.append("Shape: " + FlagToShape(shapeFlag))
    testbenchList.append("Number on Cylinder: " + str(classIndexCylinder))
    testbenchList.append("Probability (Cylinder):" + str(probValCylinder))

    testPassFail.append("------------------------------")
    testPassFail.append("AFTER COLOUR DETECTION AND CYLINDER RECOGNITION")
    testPassFail.append("------------------------------")

    predictedNumber = 99

    if probValCylinder > threshold:
        predictedNumber = classIndexCylinder

    if ((colourInFront != correctColour) or (objectType != correctObject) or (FlagToShape(shapeFlag) != correctShape) or (predictedNumber != correctNumber)):
        testPassFail.append("TEST FAILED")
    elif ((colourInFront == correctColour) and (objectType == correctObject) and (FlagToShape(shapeFlag) == correctShape) and (predictedNumber == correctNumber)):
        testPassFail.append("TEST PASSED")


    # --------------- PART OF TESTBENCH ---------------

    shapeFlag = 0       # RESET SHAPE FLAG BEFORE CIRCLE DETECTION PART
    oldObjectType = objectType          # To prevent overwriting at the circle part (TO BE REPAIRED LATER)

    # Converts frame to RGB. More suitable for showing using matplotlib. Assigns to a variable
    img_circle = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)

    # Now we need to convert to grayscale, then blur the image
    img_circle = cv2.cvtColor(img_circle, cv2.COLOR_BGR2GRAY)

    # img_circle = cv2.medianBlur(img_circle, 5)
    # img_circle = EdgepreProcessing(imgOriginal)

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

            # Identify the boundary coordinates of the circle
            # Since the boundaries could be in decimal, the ceil and floor of the values are obtained
            rectX_low = int(i[0]) - int(i[2])
            rectY_low = int(i[1]) - int(i[2])

            rectX_high = int(i[0]) + 1 - int(i[2])
            rectY_high = int(i[1]) + 1 - int(i[2])

            # Crop the frame according to the boundaries obtained earlier
            crop_img = imgOriginal.copy()[rectY_low:(rectY_high + 2 * r), rectX_low:(rectX_high + 2 * r)]

            # Draw center of circle in red
            cv2.circle(imgOriginal, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Print number of circles detected
        # print("Number of circles detected:", co)

        # Add circle count into list, then do averaging to find most likely number
        circleList.append(co)

        # Obtain the height and width of the cropped image
        crop_height, crop_width, something = crop_img.shape

        # Setting the limits of how small a cropped image is allowed to be
        if (crop_height > 100) and (crop_width > 100):
            img = np.asarray(crop_img)
            # print(img.shape)
            img2 = img.copy()
            img2 = newpreProcessing(img2)
            img = cv2.resize(img,(32,32))
            # img = preProcessing(img)
            img = newpreProcessing(img)
            cv2.imshow("Processsed Image",img2)
            img = img.reshape(1,32,32,1)
            #### PREDICT
            classIndexCircle = int(model.predict_classes(img))
            #print(classIndex)
            predictionsCircle = model.predict(img)
            #print(predictions)
            probValCircle = np.amax(predictionsCircle)
            # print(classIndex,probVal)
            shapeFlag = 1
            shape = "Sphere"

    else:
        objectType = oldObjectType

    # DISPLAY OUTCOME
    # This list stores the number of circles detected by the Hough Transform
    # For each frame, the number of circles is appended into the list
    # By checking the length of the list, we can figure out the duration of the circle/sphere being in view of the camera

    # Setting the length limit to be 20
    if len(circleList) > 20:

        circle_flag = 1
        # Reset the list for next run of circle/sphere recognition
        circleList.clear()

    # Threshold of minimum probability value for the prediction
    if probValCircle> threshold:
        # Checks if the 'guess' variable has the same value as the predicted digit
        # What this mechanism does is ensure that the predicted digits for each frame is the same consecutively
        # For example, if the first frame predicts a digit of '1'
        # When the object with a digit '1' is in view of the camera for a duration, the next frames and their predictions
        # should be '1' as well
        # When the object changes with a different digit, this digit tracking is reset, thus resetting the 'duration tracker'

        if classIndexCircle == guess:
            # If the values are the same, append the predicted value into a list
            checker.append(classIndexCircle)
        else:
            # Set the 'guess' variable as the predicted digit
            guess = classIndexCircle

        # Checking the length of the list (indicates the duration) and confirmed that a circle/sphere is in view
        if len(checker) > 15 and circle_flag == 1:
            # From the list of predicted values, select the most frequently appeared digit
            # At times a small change in ambient light or orientation may cause a single frame to predict a different digit
            # By selecting the most frequently appeared digit, it yields the most likely digit displayed on the object
            largest = checker.count(list(set(checker))[0])
            largestitem = list(set(checker))[0]

            for i in range(1, len(list(set(checker)))):
                if checker.count(list(set(checker))[i]) > largest:
                    largest = checker.count(list(set(checker))[i])
                    largestitem = list(set(checker))[i]

            # After obtaining the most frequently appeared digit, set this is as the predicted value
            classIndexCircle = largestitem

            # Display the object type in terms of the form and the digit printed on the object
            print("Mineral Type:", shape)
            print("Number on Mineral:", classIndexCircle)
            print()
            # Reset the flag indicating the object is a circle/sphere and clear the predicted values list
            circle_flag = 0
            checker.clear()

        # Show the predicted digit and probability value in the camera feed
        text1 = "Predicted: " + str(classIndexCircle)
        text2 = "Probability: " + str(probValCircle)
        text3 = "Shape: " + FlagToShape(shapeFlag)
        text4 = "Object Type: " + objectType
        text5 = "Colour: " + colourInFront
        # Display the text on the original frame (directly recorded from camera)
        cv2.putText(imgOriginal,text1,
                    (50,340),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text2,
                    (50,370),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text3,
                    (50,400),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text4,
                    (50,430),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text5,
                    (50,460),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)

    else:
        # If the probability value does not exceed the threshold, display NONE for predicted value and probability value
        text1 = "Predicted: NONE"
        text2 = "Probability: NONE"
        text3 = "Shape: " + FlagToShape(shapeFlag)
        text4 = "Object Type: " + objectType
        text5 = "Colour: " + colourInFront
        cv2.putText(imgOriginal,text1,
                    (50,340),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text2,
                    (50,370),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text3,
                    (50,400),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text4,
                    (50,430),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text5,
                    (50,460),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)

    # --------------- PART OF TESTBENCH ---------------
    # After Circle Recognition
    testbenchList.append("------------------------------")
    testbenchList.append("AFTER CIRCLE RECOGNITION")
    testbenchList.append("------------------------------")
    testbenchList.append("Colour: " + colourInFront)
    testbenchList.append("Object Type: " + objectType)
    testbenchList.append("Shape: " + FlagToShape(shapeFlag))
    testbenchList.append("Number on Circle: " + str(classIndexCircle))
    testbenchList.append("Probability (Circle): " + str(probValCircle))
    testbenchList.append("================================")
    testbenchList.append("   ")

    testPassFail.append("------------------------------")
    testPassFail.append("AFTER CIRCLE RECOGNITION")
    testPassFail.append("------------------------------")

    predictedNumber = 99

    if probValCircle > threshold:
        predictedNumber = classIndexCircle

    if ((colourInFront != correctColour) or (objectType != correctObject) or (FlagToShape(shapeFlag) != correctShape) or (predictedNumber != correctNumber)):
        testPassFail.append("TEST FAILED")
    elif ((colourInFront == correctColour) and (objectType == correctObject) and (FlagToShape(shapeFlag) == correctShape) and (predictedNumber == correctNumber)):
        testPassFail.append("TEST PASSED")
    testPassFail.append("================================")
    testPassFail.append("   ")
    # --------------- PART OF TESTBENCH ---------------

    # Show the original frame
    cv2.imshow("Original Image",imgOriginal)

    # if len(imageContainer) > 20:        # Change the number to increase or decrease the duration per image
    #     imageCounter = imageCounter + 1
    #     if imageCounter == len(imageNames):
    #         imageCounter = 0        # Restart the loop
    #         # cv2.destroyAllWindows()   # End the loop
    #         # break
    #     imageContainer.clear()
    #     cv2.destroyAllWindows()

    # ================ CAMERA FEED PART ================
    # key = cv2.waitKey(1)

    # if key & 0xFF == ord('q'):
    #     break

    # ================ CAMERA FEED PART ================

    return testbenchList, testPassFail