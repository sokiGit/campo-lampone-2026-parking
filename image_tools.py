import sys

import cv2
import numpy as np
import requests
from cv2.typing import MatLike

def extract_color_mask_min(img_hsv: MatLike, hue_min: int, hue_max: int, sat_min: int = 100, val_min: int = 20) -> MatLike:

                    #H         S       V
    min_c = np.array([hue_min, sat_min, val_min])
    max_c = np.array([hue_max, 255, 255])

    mask = cv2.inRange(img_hsv, min_c, max_c)
    return mask


def bgr_to_hsv(img_bgr: MatLike) -> MatLike:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
