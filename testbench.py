from main import imageProcessing
import os
from os import listdir
from os.path import isfile, join
import cv2
import sys

imageSkip = [
"blue_circle_9_10.jpg",
"blue_circle_9_11.jpg",
"blue_circle_9_12.jpg",
"blue_circle_9_13.jpg",
"blue_circle_9_18.jpg",
"blue_circle_9_2.jpg",
"blue_circle_9_22.jpg",
"blue_circle_9_23.jpg",
"blue_circle_9_4.jpg",
"blue_circle_9_5.jpg",
"blue_circle_9_6.jpg",
"red_circle_1_10.jpg",
"red_circle_1_11.jpg",
"red_circle_1_20.jpg",
"red_circle_1_23.jpg",
"red_circle_1_25.jpg",
"red_circle_1_26.jpg",
"red_circle_1_27.jpg",
"red_circle_1_3.jpg",
"red_circle_1_4.jpg",
"red_circle_1_6.jpg",
"red_circle_4_1.jpg",
"red_circle_4_10.jpg",
"red_circle_4_11.jpg",
"red_circle_4_12.jpg",
"red_circle_4_13.jpg",
"red_circle_4_14.jpg",
"red_circle_4_15.jpg",
"red_circle_4_22.jpg",
"red_circle_4_23.jpg",
"red_circle_4_24.jpg",
"red_circle_4_25.jpg",
"red_circle_4_27.jpg",
"red_circle_4_30.jpg",
"red_circle_4_31.jpg",
"red_circle_4_32.jpg",
"red_circle_4_33.jpg",
"red_circle_4_4.jpg",
"red_circle_4_5.jpg",
"red_circle_4_7.jpg",
"red_circle_4_8.jpg",
"red_cylinder_3_19.jpg",
"red_cylinder_3_20.jpg",
"red_cylinder_3_21.jpg"
]

imageCounter = 0                # To do a loop of running all images
imageContainer = []             # To count the number of images loaded, basically a duration counter
imageDirPath = os.path.join('images')
imageNames = [f for f in listdir(imageDirPath) if isfile(join(imageDirPath, f))]

for skips in imageSkip:
    if skips in imageNames:
        imageNames.remove(skips)

os.system("cls")
# textfile = open("testbench.txt", "w")
# textfile.close()
# textfile = open("testpassfail.txt", "w")
# textfile.close()

# testbench, testpassfail = imageProcessing("images/red_circle_4_10.jpg")
# imageProcessing("images/red_circle_4_10.jpg")

while True:
    imageContainer.append(imageNames[imageCounter])
    imageContainer.append(imageNames[imageCounter])
    imagePath = os.path.join('images', imageNames[imageCounter])
    testbench, testpassfail = imageProcessing(imagePath)
    
    # textfile = open("testbench.txt", "a")
    # for element in testbench:
    #     textfile.write(element + "\n")
    # textfile.close()

    # textfile = open("testpassfail.txt", "a")
    # for element in testpassfail:
    #     textfile.write(element + "\n")
    # textfile.close()

    key = cv2.waitKey(1)

    if len(imageContainer) > 0:        # Change the number to increase or decrease the duration per image
        imageCounter = imageCounter + 1
        if imageCounter == len(imageNames):
            imageCounter = 0        # Restart the loop
            # cv2.destroyAllWindows()   # End the loop
            # break
        imageContainer.clear()
        # cv2.destroyAllWindows()

    # Press 'q' to close the program
    if key & 0xFF == ord('q'):
        break