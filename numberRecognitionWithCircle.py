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

#### PREPORCESSING FUNCTION WITH SHOWING
def newpreProcessing(img):
    # cv2.imshow("Before Processing", img)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,0])
    upper = np.array([179,174,176])
    mask = cv2.inRange(image, lower, upper)

    mask = 255 - mask

    # cv2.imshow("The Mask", mask)

    cv2.imwrite("eBB2Q.jpg", mask)
    cv2.imwrite("luraB.jpg", image)

    mask = cv2.imread('eBB2Q.jpg')
    face = cv2.imread('luraB.jpg')

    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    # copy where we'll assign the new values
    green_hair = np.copy(face)
    # boolean indexing and assignment based on mask
    green_hair[(mask==255).all(-1)] = [255,255,255]

    green_hair = cv2.cvtColor(green_hair, cv2.COLOR_BGR2GRAY)

    # result = cv2.bitwise_and(image, image, mask=mask)
    
    # img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    # img = cv2.equalizeHist(img)

    # img = green_hair

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(img)
    
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # v = clahe.apply(v)
    # img = cv2.merge([h, s, v])

    # img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # # img = cv2.equalizeHist(img)

    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # img = cv2.inRange(img, (0, 0, 0), (70, 70, 70))

    # img = 255 - img

    # cv2.imshow("After Processing", green_hair)

    img = green_hair

    img = cv2.equalizeHist(img)

    img = img/255

    return img    

while True:
    outcome = 0

    probVal = 0

    # imgOriginal = cv2.imread('red_circle_4_2.jpg')
    # imgOriginal = cv2.imread('red_circle_1_1.jpg')        This one can't detect the circle
    # imgOriginal = cv2.imread('red_circle_1_2.jpg')
    # imgOriginal = cv2.imread('red_circle_1_7.jpg')
    # imgOriginal = cv2.imread('red_circle_1_14.jpg')
    imgOriginal = cv2.imread('blue_circle_9_20.jpg')
    # imgOriginal = cv2.imread('blue_circle_9_21.jpg')

    # success, imgOriginal = cap.read()

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

            # Identify the boundary coordinates of the circle
            # Since the boundaries could be in decimal, the ceil and floor of the values are obtained
            rectX_low = int(i[0]) - int(i[2])
            rectY_low = int(i[1]) - int(i[2])

            rectX_high = int(i[0]) + 1 - int(i[2])
            rectY_high = int(i[1]) + 1 - int(i[2])

            # Crop the frame according to the boundaries obtained earlier
            crop_img = imgOriginal[rectY_low:(rectY_high + 2 * r), rectX_low:(rectX_high + 2 * r)]

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
            classIndex = int(model.predict_classes(img))
            #print(classIndex)
            predictions = model.predict(img)
            #print(predictions)
            probVal= np.amax(predictions)
            # print(classIndex,probVal)

    # DISPLAY OUTCOME
    # This list stores the number of circles detected by the Hough Transform
    # For each frame, the number of circles is appended into the list
    # By checking the length of the list, we can figure out the duration of the circle/sphere being in view of the camera

    # Setting the length limit to be 20
    if len(circleList) > 20:
        shape = "Sphere"

        circle_flag = 1
        # Reset the list for next run of circle/sphere recognition
        circleList.clear()

    # Threshold of minimum probability value for the prediction
    if probVal> threshold:
        # Checks if the 'guess' variable has the same value as the predicted digit
        # What this mechanism does is ensure that the predicted digits for each frame is the same consecutively
        # For example, if the first frame predicts a digit of '1'
        # When the object with a digit '1' is in view of the camera for a duration, the next frames and their predictions
        # should be '1' as well
        # When the object changes with a different digit, this digit tracking is reset, thus resetting the 'duration tracker'

        if classIndex == guess:
            # If the values are the same, append the predicted value into a list
            checker.append(classIndex)
        else:
            # Set the 'guess' variable as the predicted digit
            guess = classIndex

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
            classIndex = largestitem

            # Display the object type in terms of the form and the digit printed on the object
            print("Mineral Type:", shape)
            print("Number on Mineral:", classIndex)
            print()
            # Reset the flag indicating the object is a circle/sphere and clear the predicted values list
            circle_flag = 0
            checker.clear()

        # Show the predicted digit and probability value in the camera feed
        text1 = "Predicted: " + str(classIndex)
        text2 = "Probability: " + str(probVal)
        # Display the text on the original frame (directly recorded from camera)
        cv2.putText(imgOriginal,text1,
                    (50,430),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text2,
                    (50,460),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
    else:
        # If the probability value does not exceed the threshold, display NONE for predicted value and probability value
        text1 = "Predicted: NONE"
        text2 = "Probability: NONE"
        cv2.putText(imgOriginal,text1,
                    (50,430),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text2,
                    (50,460),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)

    # Show the original frame
    cv2.imshow("Original Image",imgOriginal)

    key = cv2.waitKey(1)

    # Press 'q' to close the program
    if key & 0xFF == ord('q'):
        break