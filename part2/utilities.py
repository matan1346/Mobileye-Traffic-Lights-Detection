from typing import Tuple

import numpy as np
from .settings import *


def image_max_shape(image: np.ndarray) -> np.ndarray:
    """
    getting image array and reduce shape into SHAPE FIRST 2 dimentions
    :param image: cropped image, np.ndarray
    :return: fixed image within shape of maximum as setting.SHAPE
    """
    shape = image.shape
    if shape[0] > SHAPE[0]:
        image = image[:-(shape[0] - SHAPE[0]), :]

    if shape[1] > SHAPE[1]:
        image = image[:, :-(shape[1] - SHAPE[1])]
    return image


def is_cropped_image_ok(cropped_image: np.ndarray) -> bool:
    """
    chek whethere the cropped image is withing the shape of the settings, return true if ok, else false
    :param cropped_image: np.ndarray
    :return: bool, true if cropped image ok, else not
    """
    for i in range(len(SHAPE)):
        if SHAPE[i] != cropped_image.shape[i]:
            return False
    return True


def crop_image(img_arr: np.ndarray, pixels):
    padd_image = np.pad(img_arr, ((41, 41), (41, 41), (0, 0)), 'constant', constant_values=0)
    cropped_img = padd_image[
                  (pixels[0] - SIZE_CROP + 1): (pixels[0] + SIZE_CROP),
                  (pixels[1] - SIZE_CROP) + 1: (pixels[1] + SIZE_CROP)]

    return cropped_img


def crop_tfl_img(img_arr: np.ndarray, y_max: int, y_min: int, x_max: int, x_min: int) -> np.ndarray:
    """
    cropping traffic light image array within the specific positions given
    :param img_arr: np.nd image array
    :param y_max: int, max y position
    :param y_min: int, min y position
    :param x_max: int, max x position
    :param x_min: int, min y position
    :return: cropped image within a box around the positions
    """
    cropped_img = np.pad(img_arr, ((40, 40), (40, 40), (0, 0)), 'constant', constant_values=0)
    x_pad = (80 - (x_max - x_min)) // 2 + 1
    y_pad = (80 - (y_max - y_min)) // 2 + 1
    cropped_img = cropped_img[
                  (x_min + SIZE_CROP - x_pad): (x_max + SIZE_CROP + x_pad),
                  (y_min + 40 - y_pad):                 (y_max + 40 + y_pad)]
    return image_max_shape(cropped_img)


def crop_not_fl_img(img_arr: np.ndarray, pixels: Tuple[int, int]) -> np.ndarray:
    """
    cropping not traffic light image, within 2 pixels
    :param img_arr: np.ndarray image
    :param pixels: Tuple, y and x pixels
    :return: cropped image withing box around the pixels
    """
    cropped_img = np.pad(img_arr, ((40, 40), (40, 40), (0, 0)), 'constant', constant_values=0)
    y, x = round(pixels[0] + 40), round(pixels[1] + 40)
    cropped_img = cropped_img[
                  (y - SIZE_CROP): (y + SIZE_CROP),
                  (x - SIZE_CROP): (x + SIZE_CROP)]
    return image_max_shape(cropped_img)
