import cv2
import numpy as np

########### PARAMETERS ##############
width = 640
height = 480
#####################################

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

while True:
    imgOriginal = cv2.imread('red_circle_4_2.jpg')
    imgSharpened = unsharp_mask(imgOriginal)

    cv2.imshow("Original Image", imgOriginal)
    cv2.imshow("Sharpened Image", imgSharpened)

    key = cv2.waitKey(1)

    # Press 'q' to close the program
    if key & 0xFF == ord('q'):
        break
