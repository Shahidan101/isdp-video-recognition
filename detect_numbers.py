# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import io
import os
from google.cloud import vision_v1p3beta1 as vision
import sys

def numberRecognition(image):

	image = imutils.resize(image, height=500)

	# Grayscale the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Get the contours in the image
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	# Edged will show the contours in the image
	edged = cv2.Canny(blurred, 50, 200, 255)
	# The remaining code retrieves all the countours and sorts it from biggest to smallest
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	displayCnt = None
	# This flag is used in case there are no contours with 4 vertices (like a square/rectangle)
	flag = 0

	# This for-loop block finds the contour with 4 vertices, indicating the sign
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if the contour has four vertices, then we have found the sign
		if len(approx) == 4:
			displayCnt = approx
			# Flag is 1 indicating there is a contour with 4 vertices, so displayCnt won't be None and break the code later
			flag = 1
			break

	# If there is 4-verticed contour, run the text detection within this contour
	if flag == 1:
		warped = four_point_transform(gray, displayCnt.reshape(4, 2))
		output = four_point_transform(image, displayCnt.reshape(4, 2))

		cv2.imwrite("output.jpg", gray)

		client = vision.ImageAnnotatorClient()

		img_path = SOURCE_PATH + 'output.jpg'

		with io.open(img_path, 'rb') as image_file :
		    content = image_file.read()
		image = vision.types.Image(content=content)
		response = client.text_detection(image=image)
		texts=response.text_annotations

		empty_list = []

		for text in texts:
			stringy = text.description
			stringy = stringy.strip()
			stringy = stringy.replace('\n', '')
			for letters in stringy:
				if (letters.isnumeric()) and (letters not in empty_list):
					empty_list.append(letters)

		for things in empty_list:
			print(things, end=",")

		print()

		empty_list.clear()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'isdp-video-processing-6141e3fdd28a.json'
SOURCE_PATH = 'C:/Users/shahi/Documents/git/isdp-video-recognition/'

cam = cv2.VideoCapture(0)

while True:
	check,image1 = cam.read()

	numberRecognition(image1)

	cv2.imshow("plate detection", image1)

	key = cv2.waitKey(1)

	# Press ESC key to terminate loop
	if key == 27:
		break