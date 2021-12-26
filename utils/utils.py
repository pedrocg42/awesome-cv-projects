import os
import cv2 as cv
import numpy as np
import time
import tensorflow_hub as hub
import tensorflow as tf

FONT = cv.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.05
THICKNESS = 1
COLOR = (0, 0, 0)

#########################################################
################## GENERAL DRAWING ######################
#########################################################


def draw_text_top_left_height(
    image: np.array,
    text: str,
    height: float = 0.02,
    position: tuple = (0.04, 0.01),
    font: int = FONT,
    thickness: int = THICKNESS,
    color: tuple = COLOR,
):
    """Displays text on the top left corner of the image
    automatically calculating the font scale necessary to match
    the height (relative to the image) specify by parameter

    :param image: image to add the text to
    :type image: np.array
    :param text: text to add in the image
    :type text: str
    :param height: relative height desired for the text, defaults to 0.05
    :type height: float, optional
    :param position: relative position desired for the text (Y,X), defaults to (0.02, 0.01)
    :type position: tuple, optional
    :param font: font desired for the text, defaults to FONT
    :type font: int, optional
    :param thickness: thickness desired for the text, defaults to THICKNESS
    :type thickness: int, optional
    :param color: color desired ofr the text, defaults to COLOR
    :type color: tuple, optional
    :return: return the image with the text already drawed
    :rtype: np.array
    """

    im_height, im_width = image.shape[:2]

    # Converting relative position to absolute position in pixels
    textY = int(position[0] * im_height)
    textX = int(position[1] * im_width)

    # Converting relative height to absolute height in pixels
    height_pixels = int(height * im_height)

    # Calculating font scale to match desired height
    font_scale = cv.getFontScaleFromHeight(
        fontFace=FONT, pixelHeight=height_pixels, thickness=THICKNESS
    )

    cv.putText(
        image,
        text,
        (textX, textY),
        font,
        font_scale,
        color=color,
        thickness=thickness,
        lineType=cv.LINE_AA,
    )

    return image


#########################################################
##################### REAL TIME #########################
#########################################################
def getFPS(times):

    times[1:] = times[:-1]
    times[0] = time.time()

    intervals = np.zeros(len(times) - 1)

    for i in range(len(intervals)):
        if times[i + 1] != 0:
            intervals[i] = times[i] - times[i + 1]
        else:
            intervals[i + 1 :] = np.mean(intervals[: i + 1])
            break

    avg_interval = np.mean(intervals)

    return 1 / avg_interval if avg_interval != 0 else 0


#########################################################
################# TENSORFLOW MODELS #####################
#########################################################


def download_model_tf_hub(model_dict: dict, dir_path: str, return_model: bool = True):
    """Download the model from the TF hub in a models folder and
    return the already loaded model ready for inferente if
    return_model is True

    :param model_dict: dictionary with the information of the hub model
    :type model_dict: dict
    :param dir_path: directory path where you want to save the model
    :type dir_path: str
    :param return_model: flag to return model, defaults to True
    :type return_model: bool, optional
    """

    # Downloading and creating model
    print(f"> Downloading TF hub model from {model_dict['hub_path']}")
    model = hub.load(model_dict["hub_path"])

    # Saving model
    model_path = os.path.join(dir_path, "models", model_dict["model_path"])
    os.makedirs(model_path, exist_ok=True)
    tf.saved_model.save(model, model_path, signatures=model.signatures)
    print(f"> Model saved in {model_path}")

    if return_model:
        return model

    return True


def convert_saved_model_tflite(
    saved_model_path: str, precision: str = "float32", return_model: bool = False
):
    """Converts and saves a tflite from a saved model

    :param saved_model_path: absolute path to the saved model
    :type saved_model_path: str
    :param precision: precision desired of the output tflite model, defaults to float32
    :type precision: str, optional
    :param return_model: flag in case you want to return the created tflite model
    :type return_model: bool, optional
    """
    print(
        f"> Converting saved model in {saved_model_path} to tflite with precision {precision}"
    )
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

    # Specifying optimization and precision of
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if precision == "float32":
        converter.target_spec.supported_types = [tf.float32]
    elif precision == "float16":
        converter.target_spec.supported_types = [tf.float16]

    # Creating tflite model
    tflite_model = converter.convert()

    # Extracting output folder from saved_model and creating tflite model
    tflite_model_file = f"{saved_model_path}.tflite"
    # Save the model.
    with open(tflite_model_file, "wb") as f:
        f.write(tflite_model)
    print(f"> Converted model and saved in {tflite_model_file}")

    if return_model:
        return tflite_model

    return True


def predict_tflite_model(model: tf.lite.Interpreter, input_data: np.array):
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    if len(input_details) > 1:
        for idx, details in enumerate(input_details):
            model.set_tensor(details["index"], input_data[idx])
    else:
        if -1 in input_details[0]["shape_signature"]:
            model.resize_tensor_input(input_details[0]["index"], input_data.shape)
            model.allocate_tensors()
        model.set_tensor(input_details[0]["index"], input_data)

    model.invoke()

    if len(output_details) == 1:
        return model.get_tensor(output_details[0]["index"])
    else:
        output = {}
        for i, output_detail in enumerate(output_details):
            output[i] = model.get_tensor(output_detail["index"])

        return output


def load_tflite_model(
    model_path: str, input_shape: tuple = None, num_threads: int = None
):

    interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    random_input = None
    if len(input_details) > 1:
        random_input = []
        for details in input_details:
            random_input.append(
                np.zeros(tuple(details["shape"]), dtype=details["dtype"])
            )
    else:
        random_input = np.zeros(
            tuple(input_details[0]["shape"]), dtype=input_details[0]["dtype"]
        )

    predict_tflite_model(interpreter, random_input)

    return interpreter
