import os
import cv2 as cv
import numpy as np
import time
import tensorflow_hub as hub
import tensorflow as tf
from imutils.object_detection import non_max_suppression

FONT = cv.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.05
THICKNESS = 1
COLOR = (0, 0, 0)

#########################################################
################## GENERAL DRAWING ######################
#########################################################


def get_optimal_font_scale_height(text, height, font=FONT, thickness=THICKNESS):
    """Calculates the optimal font scale to match some height (pixels)
    required by parameter

    :param text: text that will be draw
    :type text: str
    :param height: desired height in pixels
    :type height: int
    :param font: font of the text that will be draw, defaults to FONT
    :type font: int, optional
    :param thickness: thickness of the text that will be draw, defaults to THICKNESS
    :type thickness: int, optional
    :return: font scale necessary for the text to match the desired height
    :rtype: float
    """
    for scale in reversed(np.arange(0.1, 3, 0.05)):
        new_height = cv.getTextSize(
            text, fontFace=font, fontScale=scale, thickness=thickness
        )[0][1]
        if new_height <= height:
            return scale

    return 1


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
    font_scale = get_optimal_font_scale_height(text, height=height_pixels)

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
################# OBJECT DETECTION ######################
#########################################################

# Labels from coco dataset 2017 (https://cocodataset.org/#home)
OBJECT_DETECTION_LABELS_COCO_2017 = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "street sign",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye glasses",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    45: "plate",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    66: "mirror",
    67: "dining table",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
    91: "hair brush",
}

OBJECT_DETECTION_COLORS = {
    "person": (0, 255, 0),
    "bicycle": (200, 200, 200),
    "car": (5, 138, 240),
    "motorcycle": (0, 0, 255),
    "airplane": (200, 200, 200),
    "bus": (200, 200, 200),
    "train": (200, 200, 200),
    "truck": (200, 200, 200),
    "boat": (200, 200, 200),
    "traffic light": (200, 200, 200),
    "fire hydrant": (200, 200, 200),
    "street sign": (200, 200, 200),
    "stop sign": (200, 200, 200),
    "parking meter": (200, 200, 200),
    "bench": (200, 200, 200),
    "bird": (200, 200, 200),
    "cat": (117, 138, 26),
    "dog": (200, 200, 200),
    "horse": (200, 200, 200),
    "sheep": (200, 200, 200),
    "cow": (200, 200, 200),
    "elephant": (200, 200, 200),
    "bear": (200, 200, 200),
    "zebra": (200, 200, 200),
    "giraffe": (200, 200, 200),
    "hat": (200, 200, 200),
    "backpack": (200, 200, 200),
    "umbrella": (200, 200, 200),
    "shoe": (200, 200, 200),
    "eye glasses": (200, 200, 200),
    "handbag": (200, 200, 200),
    "tie": (200, 200, 200),
    "suitcase": (200, 200, 200),
    "frisbee": (200, 200, 200),
    "skis": (200, 200, 200),
    "snowboard": (200, 200, 200),
    "sports ball": (200, 200, 200),
    "kite": (200, 200, 200),
    "baseball bat": (200, 200, 200),
    "baseball glove": (200, 200, 200),
    "skateboard": (200, 200, 200),
    "surfboard": (200, 200, 200),
    "tennis racket": (200, 200, 200),
    "bottle": (200, 200, 200),
    "plate": (200, 200, 200),
    "wine glass": (200, 200, 200),
    "cup": (200, 200, 200),
    "fork": (200, 200, 200),
    "knife": (200, 200, 200),
    "spoon": (200, 200, 200),
    "bowl": (200, 200, 200),
    "banana": (200, 200, 200),
    "apple": (200, 200, 200),
    "sandwich": (200, 200, 200),
    "orange": (200, 200, 200),
    "broccoli": (200, 200, 200),
    "carrot": (200, 200, 200),
    "hot dog": (200, 200, 200),
    "pizza": (200, 200, 200),
    "donut": (200, 200, 200),
    "cake": (200, 200, 200),
    "chair": (200, 200, 200),
    "couch": (200, 200, 200),
    "potted plant": (200, 200, 200),
    "bed": (200, 200, 200),
    "mirror": (200, 200, 200),
    "dining table": (200, 200, 200),
    "window": (200, 200, 200),
    "desk": (200, 200, 200),
    "toilet": (200, 200, 200),
    "door": (200, 200, 200),
    "tv": (200, 200, 200),
    "laptop": (200, 200, 200),
    "mouse": (200, 200, 200),
    "remote": (200, 200, 200),
    "keyboard": (200, 200, 200),
    "cell phone": (200, 200, 200),
    "microwave": (200, 200, 200),
    "oven": (200, 200, 200),
    "toaster": (200, 200, 200),
    "sink": (200, 200, 200),
    "refrigerator": (200, 200, 200),
    "blender": (200, 200, 200),
    "book": (200, 200, 200),
    "clock": (200, 200, 200),
    "vase": (200, 200, 200),
    "scissors": (200, 200, 200),
    "teddy bear": (200, 200, 200),
    "hair drier": (200, 200, 200),
    "toothbrush": (200, 200, 200),
    "hair brush": (200, 200, 200),
}

# TODO: do my own version of nms
def non_max_suppression(scores, boxes, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = scores

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    # return only the bounding boxes that were picked
    return pick


def draw_object_detection_box(image, box, score, class_name, box_format: int = 0):

    im_height, im_width = image.shape[:2]

    # correcting boxes out of the image
    box = box.clip(0, 1)

    # extracting bounding box
    if box_format == 0:
        x1 = (box[1] * im_width).astype(int)
        y1 = (box[0] * im_height).astype(int)
        x2 = (box[3] * im_width).astype(int)
        y2 = (box[2] * im_height).astype(int)
    elif box_format == 1:
        x1 = (box[1] * im_width).astype(int)
        y1 = (box[1] * im_height).astype(int)
        x2 = ((box[1] + box[3]) * im_width).astype(int)
        y2 = ((box[0] + box[2]) * im_height).astype(int)

    # draw a bounding box rectangle and label on the image
    color = OBJECT_DETECTION_COLORS[class_name]
    cv.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Creating text and calculating size
    text = f"{class_name} - {score:.2f}"
    text_height = int(0.015 * im_height)
    font_scale = cv.getFontScaleFromHeight(
        fontFace=FONT, pixelHeight=text_height, thickness=THICKNESS
    )

    # showing text and background
    text_padding = int(im_height * 0.008)
    text_width = cv.getTextSize(
        text, fontFace=FONT, fontScale=font_scale, thickness=THICKNESS
    )[0][0]
    text_x = x1 + text_padding
    if y1 - text_height - 2 * text_padding >= 0:
        text_y = y1 - text_padding
        cv.rectangle(
            image,
            (x1, y1 - text_height - 2 * text_padding),
            (x1 + text_width + 2 * text_padding, y1),
            color,
            -1,
        )
    else:
        text_y = y1 + text_height + text_padding
        cv.rectangle(
            image,
            (x1, y1),
            (x1 + text_width + 2 * text_padding, y1 + text_height + 2 * text_padding),
            color,
            -1,
        )

    # showing background box for text
    # showing text
    cv.putText(
        image,
        text,
        (text_x, text_y),
        cv.FONT_HERSHEY_COMPLEX,
        font_scale,
        COLOR,
        THICKNESS,
        lineType=cv.LINE_AA,
    )

    return image


def draw_raw_object_detection_output(
    image: np.array,
    raw_scores: np.array,
    raw_classes: np.array,
    raw_boxes: np.array,
    threshold: float = 0.2,
    class_dict: dict = OBJECT_DETECTION_LABELS_COCO_2017,
    non_max: bool = True,
    nms_threshold: float = 0.65,
):

    # extracting the scores and classes from the raw scores
    # raw_scores = np.max(raw_scores_class, axis=1)
    # raw_classes = np.argmax(raw_scores_class, axis=1)

    # extracting the scores and classes that are
    # higher than the defined threshold
    scores = raw_scores[raw_scores > threshold]
    classes = raw_classes[raw_scores > threshold]
    boxes = raw_boxes[raw_scores > threshold]

    # non max supression applied to the detections if required
    if non_max:
        unique_classes = np.unique(classes)
        for unique_class in unique_classes:
            index_class = np.where(classes == unique_class)[0]
            # only apply nms if we detect more than object of the specific class
            if len(index_class) > 1:
                pick = non_max_suppression(
                    scores[index_class], boxes[index_class], overlapThresh=0.5
                )
                delete_idxs = np.delete(index_class, pick)
                # deleting overlapping detections
                scores = np.delete(scores, delete_idxs)
                classes = np.delete(classes, delete_idxs)
                boxes = np.delete(boxes, delete_idxs, axis=0)

    # drawing box for each
    for i, score in enumerate(scores):
        box = boxes[i]
        clas = classes[i]
        image = draw_object_detection_box(image, box, score, class_dict[int(clas)])

    return image


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
    tf.saved_model.save(model, model_path)
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

    return model.get_tensor(output_details[0]["index"])


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
