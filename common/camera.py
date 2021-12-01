import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2 as cv
from utils import plot_text_top_left_height, getFPS

# Selecting our camera
device = 0
cam = cv.VideoCapture(device)

# Setting the height of the image that we expect
cam.set(3, 1280)
cam.set(4, 720)

# Creating window to display
cv.namedWindow("Camera", cv.WINDOW_NORMAL)

# Setting FPS buffer
BUFFER_SIZE = 100
times = np.zeros(BUFFER_SIZE)

while True:
    ret, image = cam.read()

    if ret:
        # Getting and printing FPS
        fps = getFPS(times)
        plot_text_top_left_height(image, f"FPS: {fps:.2f}")

        # Showing image
        cv.imshow("Camera", image)

    key = cv.waitKey(1)
    if key == 27:
        break


cv.destroyAllWindows()
cam.release()
