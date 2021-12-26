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
    predict_tflite_model,
    draw_text_top_left_height,
    draw_skel_and_kp,
    unprocess_keypoint_coords,
)

# load tflite model (press 'l' to switch)
tflite = False

# Selecting our camera
device = 0
cam = cv.VideoCapture(device)

# Setting the height of the image that we expect
cam.set(3, 1280)
cam.set(4, 720)

# Creating window to display
cv.namedWindow("Object Detection", cv.WINDOW_NORMAL)

# Setting FPS buffer
BUFFER_SIZE = 100
times = np.zeros(BUFFER_SIZE)

model_idx = 0
models = [
    # Load this model pressing number 1 in your keyboard
    {
        "name": "singlepose-lightning",
        "model_path": "movenet_singlepose_lightning_4",
        "hub_path": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        "input_shape": (192, 192),
        "threshold": 0.5,
    },
    # Load this model pressing number 2 in your keyboard, load  by default
    {
        "name": "singlepose-thunder",
        "model_path": "movenet_singlepose_thunder_4",
        "hub_path": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
        "input_shape": (256, 256),
        "threshold": 0.5,
    },
    # Load this model pressing number 3 in your keyboard, load  by default
    {
        "name": "multipose-lightning",
        "model_path": "movenet_multipose_lightning_4",
        "hub_path": "https://tfhub.dev/google/movenet/multipose/lightning/1",
        "input_shape": (256, 256),
        "threshold": 0.5,
    },
]
load_model = True

while True:

    if load_model:
        # restarting buffer for fps
        times = np.zeros(BUFFER_SIZE)
        # initializing models variables
        input_shape = models[model_idx]["input_shape"]
        saved_model_path = os.path.join(
            dir_path, "models", models[model_idx]["model_path"]
        )
        if not tflite:
            print("Loading saved_model...")
            # if a model is already downloaded it loads it from local, if not from TF hub
            if not os.path.isdir(saved_model_path):
                model = download_model_tf_hub(models[model_idx], dir_path=dir_path)
            else:
                model = tf.saved_model.load(saved_model_path)
            movenet = model.signatures["serving_default"]
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
            # loading tflite model
            tflite_model = load_tflite_model(tflite_model_path)

        # Turn off flag to load a new model
        load_model = False
        print("Model loaded and ready for inference! Let's go!")

    ret, image = cam.read()

    if ret:
        # preprocessing image
        inference_image = tf.expand_dims(image, axis=0)
        inference_image = tf.cast(
            tf.image.resize(inference_image, input_shape), dtype=tf.int32
        )

        # inference
        if not tflite:
            output = movenet(inference_image)
        else:
            output = predict_tflite_model(tflite_model, inference_image)

        # unprocessing output
        output = tf.squeeze(output["output_0"], axis=0).numpy()
        instance_scores = np.ones(len(output))
        keypoint_scores = output[:, :, 2]
        keypoint_coords = output[:, :, :2]
        keypoint_coords = unprocess_keypoint_coords(
            keypoint_coords, image.shape[:2], input_shape
        )

        # plotting results in image
        image = draw_skel_and_kp(
            image,
            instance_scores=instance_scores,
            keypoint_scores=keypoint_scores,
            keypoint_coords=keypoint_coords,
            min_pose_score=0.5,
            min_part_score=0.5,
        )

        # Getting and printing FPS
        fps = getFPS(times)
        draw_text_top_left_height(image, f"FPS: {fps:.2f}")

        # Showing image
        cv.imshow("Object Detection", image)

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
    elif key == ord("5"):
        if model_idx != 2:
            model_idx = 4
            load_model = True
    elif key == ord("l"):
        tflite = not tflite
        load_model = True
    elif key == ord("s"):
        non_max = not non_max

cv.destroyAllWindows()
cam.release()
