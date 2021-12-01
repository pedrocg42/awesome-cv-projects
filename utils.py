import cv2 as cv
import numpy as np
import time

FONT = cv.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.05
THICKNESS = 1
COLOR = (0, 0, 0)

def get_optimal_font_scale_height(text, height, font=FONT, thickness=THICKNESS):
    """Calculates the optimal font scale to match some height (pixels)
    required by parameter

    :param text: text that will be plot
    :type text: str
    :param height: desired height in pixels
    :type height: int
    :param font: font of the text that will be plot, defaults to FONT
    :type font: int, optional
    :param thickness: thickness of the text that will be plot, defaults to THICKNESS
    :type thickness: int, optional
    :return: font scale necessary for the text to match the desired height
    :rtype: float
    """
    for scale in reversed(np.arange(0.1, 3, 0.05)):
        new_height = cv.getTextSize(text, fontFace=font, fontScale=scale, thickness=thickness)[0][1]
        if (new_height <= height):
            return scale
    
    return 1

def plot_text_top_left_height(
    image: np.array,
    text: str,
    height: float=0.02,
    position: tuple=(0.04, 0.01),
    font: int=FONT,
    thickness: int=THICKNESS,
    color: tuple=COLOR,
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
    :return: return the image with the text already plotted
    :rtype: np.array
    """

    im_height, im_width = image.shape[:2]

    # Converting relative position to absolute position in pixels
    textY = int(position[0] * im_height)
    textX = int(position[1] * im_width)

    # Converting relative height to absolute height in pixels
    height_pixels = int(height*im_height)

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


def getFPS(times):

    times[1:] = times[:-1]
    times[0] = time.time()

    intervals = np.zeros(len(times)-1)

    for i in range(len(intervals)):
        if times[i+1] != 0:
            intervals[i] = times[i] - times[i+1]
        else:
            intervals[i+1:] = np.mean(intervals[:i+1])
            break

    avg_interval = np.mean(intervals)

    return 1 / avg_interval