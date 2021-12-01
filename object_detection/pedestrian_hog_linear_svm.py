"""The following implementation uses the already trained linear svm
which uses the features extracted by the Histogram Oriented Gradientes
(HOG) algorithm to detect pedestrians.

In this script we are going to open a camera and execute this pedestrian
detector to test how good and fast it runs in real time
"""
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2 as cv
import numpy as np
from imutils.object_detection import non_max_suppression
from utils import plot_text_top_left_height, getFPS

# Selecting our camera
device = 0
cam = cv.VideoCapture(device)

# Setting the height of the image that we expect
cam.set(3, 1280)
cam.set(4, 720)

# Setting FPS buffer
BUFFER_SIZE = 50
times = np.zeros(BUFFER_SIZE)

# Creating window to display
cv.namedWindow("Camera", cv.WINDOW_NORMAL)

# initialize the HOG descriptor/person detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, image = cam.read()

    if ret:
        height, width = image.shape[:2]

        # Getting and printing FPS
        fps = getFPS(times)
        plot_text_top_left_height(image, f"FPS: {fps:.2f}")

        # To reduce computational cost we resize the image
        REDUCE_RATIO = 3
        image_down = cv.resize(image, (0, 0), fx=1 / REDUCE_RATIO, fy=1 / REDUCE_RATIO)

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(
            image_down, winStride=(4, 4), padding=(8, 8), scale=1.05
        )

        # post process predicted rects to match image
        rects = rects * REDUCE_RATIO
        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        if len(rects):
            print("")
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # Showing image
        cv.imshow("Camera", image)

    key = cv.waitKey(1)
    if key == 27:
        break


cv.destroyAllWindows()
cam.release()
