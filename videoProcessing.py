def circleDetection():
	
	# Import necessary packages
	import numpy as np 
	import matplotlib.pyplot as plt 
	import cv2

	# Initialise Circle counter
	circleList = []

	# Define Video Capture variable
	cap = cv2.VideoCapture(0)

	while True:
		outcome = 0

		# Reads frame from video capture
		check, image = cap.read()

		# Converts frame to RGB. More suitable for showing using matplotlib. Assigns to a variable
		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Make a copy of the original image
		cimg = img.copy()

		# Now we need to convert to grayscale, then blur the image
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		img = cv2.medianBlur(img, 5)

		# Find circles int he grayscale frame using Hough Transform
		circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=0.9, minDist=80, param1=110, param2=39, minRadius=150 ,maxRadius=1000)

		if circles is not None:
			# Now draw and show the circles that were detected
			for co, i in enumerate(circles[0, :], start=1):
				# Draw outer circle in green
				cv2.circle(cimg, (i[0], i[1]), int(i[2]), (0, 255, 0), 2)

				# Draw center of circle in red
				cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

			# Print number of circles detected
			# print("Number of circles detected:", co)

			# Add circle count into list, then do averaging to find most likely number
			circleList.append(co)
		

		# Print out outcome
		if len(circleList) > 50:
			shape = "circle"
			print(shape)
			circleList.clear()

		# Convert image back to RGB
		cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR)		
		# Show the frame
		cv2.imshow("Camera Feed", cimg)

		if cv2.waitKey(1) == ord('q'):
			break

	cap.release()

def rectangleDetection():
	
	# Import necessary packages
	import numpy as np 
	import matplotlib.pyplot as plt 
	import cv2

	font = cv2.FONT_HERSHEY_COMPLEX

	# Define Video Capture variable
	cap = cv2.VideoCapture(0)

	while True:
		# Reads frame from video capture
		check, image = cap.read()

		# Get threshold of image to get black/white image
		img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		_, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_OTSU)

		# Find contours from black/white image
		contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# Loop through contours. Get coordinates of contours of each shape
		for cnt in contours:
			# Approximate contours to remove noiseq
			approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
			cv2.drawContours(img, [approx], 0, (0), 5)
			x = approx.ravel()[0]
			y = approx.ravel()[1]

			if len(approx) == 4:
				cv2.putText(img, "Rectangle", (x, y), font, 1, (0))

		cv2.imshow("Camera Feed", img)
		cv2.imshow("Threshold", threshold)

		if cv2.waitKey(1) == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

def colourDetection():
	# Import necessary packages
	import numpy as np 
	import matplotlib.pyplot as plt 
	import cv2

	# Define the list of boundaries
	boundaries = [
		([17, 15, 100], [50, 56, 200]),
		#([86, 31, 4], [220, 88, 50])
	]

	# Define Video Capture variable
	cap = cv2.VideoCapture(0)

	while True:
		# Reads frame from video capture
		check, image = cap.read()

		# Loop over boundaries
		for (lower, upper) in boundaries:
			# Create Numpy arrays from boundaries
			lower = np.array(lower, dtype="uint8")
			upper = np.array(upper, dtype="uint8")

			# Find colours within specified boundaries and apply mask
			mask = cv2.inRange(image, lower, upper)
			output = cv2.bitwise_and(image, image, mask=mask)

		# Show images
		cv2.imshow("Camera Feed", np.hstack([image, output]))

		if cv2.waitKey(1) == ord('q'):
			break

circleDetection()