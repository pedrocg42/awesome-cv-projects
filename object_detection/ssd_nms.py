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

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
CONFIDENCE = 0.5

# load our serialized model from disk
print("[INFO] Loading model...")
net = cv.dnn.readNetFromCaffe(
    "object_detection/models/MobileNetSSD_deploy.prototxt.txt",
    "object_detection/models/MobileNetSSD_deploy.caffemodel",
)

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

while True:
    ret, image = cam.read()

    if ret:
        height, width = image.shape[:2]

        # Reinitializing variables
        probs = np.empty(0)
        idxs = np.empty(0, dtype=int)
        boxes = np.empty((0, 4))

        # Getting and printing FPS
        fps = getFPS(times)
        plot_text_top_left_height(image, f"FPS: {fps:.1f}")

        # Preprocess image and inference
        blob = cv.dnn.blobFromImage(
            cv.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        detections = net.forward()

        # Postprocess detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > CONFIDENCE:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                probs = np.append(probs, confidence)
                idxs = np.append(idxs, int(detections[0, 0, i, 1]))
                boxes = np.append(
                    boxes,
                    detections[0, 0, i, 3:7]
                    * np.array([[width, height, width, height]]),
                    axis=0,
                )

        # Apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        boxes = non_max_suppression(boxes, probs=probs, overlapThresh=0.65)

        # draw the final bounding boxes
        for i, box in enumerate(boxes):
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = f"{CLASSES[idxs[i]]}: {probs[i] * 100:.2f}%"
            cv.rectangle(image, (startX, startY), (endX, endY), COLORS[idxs[i]], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv.putText(
                image,
                label,
                (startX, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS[idxs[i]],
                2,
            )

        # Showing image
        cv.imshow("Camera", image)

    key = cv.waitKey(1)
    if key == 27:
        break


cv.destroyAllWindows()
cam.release()
