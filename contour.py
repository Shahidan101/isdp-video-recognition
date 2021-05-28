import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

def detect_shape(image_path):
    image = cv2.imread(image_path)
    #resized = imutils.resize(image, width=300)
    #ratio = image.shape[0] / float(resized.shape[0])
    # Convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('grey image', image)
    # Apply gausian blur to remove high frequency noise at background
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('blurred', blurred)
    #cv2.imshow('blurred image', blurred)
    # If image later doesn't work, change to THRESH_BINARY
    # background should be blackened in order to detect the shape
    thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)[1]
    #thresh_bright = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY)[1]
    #thresh_bright = cv2.erode(thresh_bright, None, iterations=2)
    #thresh_bright = cv2.dilate(thresh_bright, None, iterations=8)
    #blurred = cv2.inpaint(image, thresh, 3, flags=cv2.INPAINT_TELEA)
    #full_thresh = thresh + thresh_bright
    #full_thresh = cv2.erode(full_thresh, None, iterations=2)
    #full_thresh = cv2.dilate(full_thresh, None, iterations=8)
    #blob = cv2.bitwise_and(blurred, blurred, mask=full_thresh)
    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,91, 30)
    #cv2.imshow('blurred image', blurred)
    #cv2.imshow('thresh image', thresh)
    #cv2.imshow('thresh bright image', thresh_bright)
    #cv2.imshow('thresh full image', full_thresh)
    #cv2.imshow('blob image', blob)
    #cv2.waitKey()
    # Find contours in the thresholded image and initialize the shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #print('contours', cnts)
    #loop for largest array of cnts
    #print(len(cnts[1]))
    print(cnts)
    for c in cnts:
        # Compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        #cX = int((M["m10"] / M["m00"]) * ratio)
        #cY = int((M["m01"] / M["m00"]) * ratio)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        shape = ("unidentified")
        peri = cv2.arcLength(c, True)
        print('perimeter', peri)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        print('length of approx', len(approx))
        # If the shape is a triangle, it will have 3 vertices
        #if len(approx) == 3:
            #shape = "triangle"
        # If the shape has 4 vertices, it is either a square or
        # a rectangle
        if len(approx) == 4:
            # Compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # A square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = ("square") if ar >= 0.95 and ar <= 1.05 else ("cylinder")
        # If the shape is a pentagon, it will have 5 vertices
        #elif len(approx) == 5:
            #shape = "pentagon"
        # Otherwise, we assume the shape is a circle
        else:#elif len(approx) == 0:
            shape = ("circle")
        #else:
            #shape = ("unknown")
        # Return the name of the shape
        # Multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        #c = c.astype("float")
        #c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Show the output image
        print(shape)
        cv2.imshow("Image", image)
        
    return shape

image = cv2.imread('/home/pi/Desktop/colourblobs-0.jpeg')
image_ori = image.copy()
cv2.imshow('image', image)
if image is None:
    print("Failed to load image.")
    exit(-1)
image_blurred = cv2.GaussianBlur(image, (3,3), 0)
image_hsv = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)

# Hue describes the color, Saturation describes greyness 
# (low saturation, high grey), Value describes brightness
# Initialize HSV limits of respective colours
#if color_str == 'RED':
lower1 = np.array([0, 150, 50])
upper1 = np.array([3, 255, 255])
lower2 = np.array([160,150,50])
upper2 = np.array([179,255,255])
#elif color_str == 'BLUE':
#lower1 = np.array([110,100,20])
#upper1 = np.array([119,255,255])
#lower2 = np.array([120,100,20])
#upper2 = np.array([139,255,255])
#elif color_str == 'GREEN':
#lower1 = np.array([60,100,20])
#upper1 = np.array([69,255,255])
#lower2 = np.array([70,100,20])
#upper2 = np.array([79,255,255])

#  Creating mask of areas with base station colour
lower_mask = cv2.inRange(image_hsv, lower1, upper1)
#cv2.imshow('lower mask', lower_mask)
upper_mask = cv2.inRange(image_hsv, lower2, upper2)
#cv2.imshow('upper mask', upper_mask)
full_mask = lower_mask + upper_mask
cv2.imshow('full_mask', full_mask)
# And use it to extract the corresponding part of the original colour image
blob = cv2.bitwise_and(image, image, mask=full_mask)

contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for j, contour in enumerate(contours):
    #print(contour)
    bbox = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    
    if area > 100:
        #print('box area', area)
        # Create a mask for this contour
        contour_mask = np.zeros_like(full_mask)
        cv2.drawContours(contour_mask, contours, j, 255, -1)
        #print('bounding box', bbox)
        # Extract and save the area of the contour
        region = blob.copy()[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        region_mask = contour_mask[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        region_masked = cv2.bitwise_and(region, region, mask=region_mask)
        #file_name_section = "colourblobs-%d-hue_%03d-region_%d-section.png" % (i, peak, j)
        #cv2.imwrite(file_name_section, region_masked)
        #print (" * wrote '%s'" % file_name_section)

        # Extract the pixels belonging to this contour
        result = cv2.bitwise_and(blob, blob, mask=contour_mask)
        # And draw a bounding box
        top_left, bottom_right = (bbox[0]-10, bbox[1]-10), (bbox[0]+bbox[2]+10, bbox[1]+bbox[3]+10)
        #print('top left', top_left)
        #print('bottom right', bottom_right)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 0)
        file_name_bbox = "colourblobs-%d.jpeg" % (j)
        #cv2.imwrite(file_name_bbox, image_ori[bbox[1]-10:bbox[1]+bbox[3]+10,bbox[0]-10:bbox[0]+bbox[2]+10])
        detect_shape(file_name_bbox)
        #print(shape)
        #print (" * wrote '%s'" % file_name_bbox)
        #plt.imshow(result),plt.show()

cv2.imshow('result', image)
cv2.waitKey()

