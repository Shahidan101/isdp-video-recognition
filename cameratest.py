import cv2

cameraNo = 0
cap = cv2.VideoCapture(cameraNo)

while True:
    width = 640
    height = 480
    cap.set(3,width)
    cap.set(4,height)
    success, imgOriginal = cap.read()

    cv2.imshow("Frame", imgOriginal)
    key = cv2.waitKey(33)

    if key == ord('q'):
        break