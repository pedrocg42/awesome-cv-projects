"""The purpose of this script is to compare the performance and
accuracy of possible object detection models for real time inference
in a normal computer cpu. We are going to compare several models 
selected from the object detection collection of TF hub 
(https://tfhub.dev/tensorflow/collections/object_detection/1)
"""
import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dir_path))

import cv2 as cv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from utils import (
    convert_saved_model_tflite,
    download_model_tf_hub,
    getFPS,
    load_tflite_model,
    plot_text_top_left_height,
    predict_tflite_model,
)

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

model_idx = 1
models = [
    # Load this model pressing number 1 in your keyboard
    {
        "name": "centernet512",
        "model_path": "centernet_resnet50v1_fpn_512x512",
        "hub_path": "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1",
        "input_shape": (512, 512),
    },
    # Load this model pressing number 2 in your keyboard, load  by default
    {
        "name": "ssd320",
        "model_path": "ssd_mobilenet_v2_fpnlite_320x320_1",
        "hub_path": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1",
        "input_shape": (320, 320),
    },
    # Load this model pressing number 3 in your keyboard
    {
        "name": "ssd640",
        "model_path": "ssd_mobilenet_v2_fpnlite_640x640_1",
        "hub_path": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1",
        "input_shape": (640, 640),
    },
    # Load this model pressing number 4 in your keyboard
    {
        "name": "faster-rcnn",
        "model_path": "faster_rcnn_resnet50_v1_640x640",
        "hub_path": "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1",
        "input_shape": (640, 640),
    },
]
load_model = True
tflite = True

while True:

    if load_model:
        # restarting buffer for fps
        times = np.zeros(BUFFER_SIZE)
        input_shape = models[model_idx]["input_shape"]
        saved_model_path = os.path.join(
            dir_path, "models", models[model_idx]["model_path"]
        )
        if not tflite:
            print("Loading saved_model...")
            # If a model is already downloaded it loads it from local, if not from TF hub
            if not os.path.isdir(saved_model_path):
                model = download_model_tf_hub(models[model_idx], dir_path=dir_path)
            else:
                model = tf.saved_model.load(saved_model_path)
        else:
            print("Loading tflite model...")
            tflite_model_path = f"{saved_model_path}.tflite"
            # if tflite model is not created we need to create it
            if not os.path.isfile(tflite_model_path):
                # if saved model is not downloaded from tf hub we need to do so
                if not os.path.isdir(saved_model_path):
                    download_model_tf_hub(
                        models[model_idx],
                        dir_path=dir_path,
                        return_model=False,
                    )
                # converting model to tflite
                convert_saved_model_tflite(saved_model_path)

            tflite_model = load_tflite_model(tflite_model_path)

        # Turn off flag to load a new model
        load_model = False
        print("Model loaded and ready for inference! Let's go!")

    ret, image = cam.read()

    if ret:
        # preprocessing image
        inference_image = cv.resize(
            image, input_shape, interpolation=cv.INTER_CUBIC
        ).reshape((1, input_shape[0], input_shape[1], 3))

        # inference
        if not tflite:
            output = model(inference_image)
        else:
            predict_tflite_model(tflite_model, inference_image)

        # Getting and printing FPS
        fps = getFPS(times)
        plot_text_top_left_height(image, f"FPS: {fps:.2f}")

        # Showing image
        cv.imshow("Camera", image)

    key = cv.waitKey(1)
    # Press esc to stop the execution
    if key == 27:
        break
    elif key == ord("1"):
        if model_idx != 0:
            model_idx = 0
            load_model = True
    elif key == ord("2"):
        if model_idx != 1:
            model_idx = 1
            load_model = True
    elif key == ord("3"):
        if model_idx != 2:
            model_idx = 2
            load_model = True
    elif key == ord("4"):
        if model_idx != 3:
            model_idx = 3
            load_model = True
    elif key == ord("l"):
        tflite = not tflite
        load_model = True

cv.destroyAllWindows()
cam.release()
