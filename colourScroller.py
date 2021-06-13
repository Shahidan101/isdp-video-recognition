import cv2
import sys
import numpy as np

# cameraNo = 0
width = 640
height = 480

# cap = cv2.VideoCapture(cameraNo)
# cap.set(3,width)
# cap.set(4,height)

def nothing(x):
    pass

# Create a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# cv2.createTrackbar('RMin','image',0,255,nothing)
# cv2.createTrackbar('GMin','image',0,255,nothing)
# cv2.createTrackbar('BMin','image',0,255,nothing)
# cv2.createTrackbar('RMax','image',0,255,nothing)
# cv2.createTrackbar('GMax','image',0,255,nothing)
# cv2.createTrackbar('BMax','image',0,255,nothing)

# Set default value for MIN HSV trackbars.
cv2.setTrackbarPos('HMin', 'image', 50)
cv2.setTrackbarPos('SMin', 'image', 146)
cv2.setTrackbarPos('VMin', 'image', 0)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 76)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Set default value for MAX RGB trackbars.
# cv2.setTrackbarPos('RMax', 'image', 255)
# cv2.setTrackbarPos('GMax', 'image', 255)
# cv2.setTrackbarPos('BMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# Initialize to check if RGB min/max value changes
# rMin = gMin = bMin = rMax = gMax = bMax = 0
# prMin = pgMin = pbMin = prMax = pgMax = pbMax = 0

# Load in image
while True:

    # success, image = cap.read()
    # image = cv2.imread('red_circle_4_2.jpg')
    # image = cv2.imread('blue_circle_9_20.jpg')
    # image = cv2.imread('testImage.jpg')
    image = cv2.imread('images/green_obstacle_99_1.jpg')
    # print(image.shape)
    # image = cv2.resize(image,(320,240))
    
    output = image
    wait_time = 33

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    # rMin = cv2.getTrackbarPos('RMin','image')
    # gMin = cv2.getTrackbarPos('GMin','image')
    # bMin = cv2.getTrackbarPos('BMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # rMax = cv2.getTrackbarPos('RMax','image')
    # gMax = cv2.getTrackbarPos('GMax','image')
    # bMax = cv2.getTrackbarPos('BMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Set minimum and max RGB values to display
    # lower = np.array([rMin, gMin, bMin])
    # upper = np.array([rMax, gMax, bMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image,image, mask= mask)

    # Create RGB Image and threshold into a range.
    # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask = cv2.inRange(rgb, lower, upper)
    # output = cv2.bitwise_and(image,image, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Print if there is a change in RGB value
    # if( (prMin != rMin) | (pgMin != gMin) | (pbMin != bMin) | (prMax != rMax) | (pgMax != gMax) | (pbMax != bMax) ):
    #     print("(rMin = %d , gMin = %d, bMin = %d), (rMax = %d , gMax = %d, bMax = %d)" % (rMin , gMin , bMin, rMax, gMax , bMax))
    #     prMin = rMin
    #     pgMin = gMin
    #     pbMin = bMin
    #     prMax = rMax
    #     pgMax = gMax
    #     pbMax = bMax

    # Display output image
    cv2.imshow('image',output)

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

    # cv2.destroyAllWindows()