import numpy as np
import argparse
import imutils
import cv2

# Opening an image
# img = cv2.imread('Pikachew.jpg', 1)          # param. 1 = read as full color, set 0 for grayscale
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# -------------------------------------------------------------------------------------------------


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

shapesImg = cv2.imread(args["image"])
gray = cv2.cvtColor(shapesImg, cv2.COLOR_BGR2GRAY)                      # conversion to grayscale
blurred = cv2.GaussianBlur(gray, (5, 5), 0)                         # Blurring to reduce HF noise
# Binarization of the image. Typically edge detection and thresholding are used for this process.
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# finding contours in the threshold image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the contour and center of the shape on the image
    cv2.drawContours(shapesImg, [c], -1, (0, 255, 0), 2)
    cv2.circle(shapesImg, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(shapesImg, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

    cv2.imshow("Image", shapesImg)
    cv2.waitKey(0)