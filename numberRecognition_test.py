import numpy as np
import cv2
import pickle
import tensorflow as tf
import time
 
########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.98 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
#####################################
 
#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0
 
checker = []

guess = 50

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
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1,32,32,1)
    #### PREDICT
    classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal= np.amax(predictions)
    # print(classIndex,probVal)

    if probVal> threshold:
        if classIndex == guess:
            checker.append(classIndex)
        else:
            checker = []
            guess = classIndex

        if len(checker) > 20:
            print(classIndex)
            checker = []
        text1 = "Predicted: " + str(classIndex)
        text2 = "Probability: " + str(probVal)
        cv2.putText(imgOriginal,text1,
                    (50,430),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text2,
                    (50,460),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
    else:
        checker = []
        text1 = "Predicted: NONE"
        text2 = "Probability: NONE"
        cv2.putText(imgOriginal,text1,
                    (50,430),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
        cv2.putText(imgOriginal,text2,
                    (50,460),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)
 
    # time when we finish processing for this frame 
    new_frame_time = time.time()

    # fps will be number of frame processed in given time frame 
    # since their will be most of time error of 0.001 second 
    # we will be subtracting it to get more accurate result 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time

    # converting the fps into integer 
    fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps)

    text3 = "FPS: " + fps
    
    cv2.putText(imgOriginal,text3,
                    (50,50),cv2.FONT_HERSHEY_DUPLEX,
                    1,(255,0,0),1)

    cv2.imshow("Original Image",imgOriginal)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break