import sys

import cv2
import numpy as np
from numpy.ma.core import MaskType
import requests
from cv2.typing import MatLike

def extract_color_mask_min(img_hsv: MatLike, hue_min: int, hue_max: int, sat_min: int = 100, val_min: int = 20) -> MatLike:

                    #H         S       V
    min_c = np.array([hue_min, sat_min, val_min])
    max_c = np.array([hue_max, 255, 255])

    mask = cv2.inRange(img_hsv, min_c, max_c)
    return mask

class MaskBoundingBox:
    def __init__(self, from_x: int, from_y: int, to_x: int, to_y: int) -> None:
        self.from_pos = {
            'x': from_x,
            'y': from_y
        }
        self.to_pos = {
            'x': to_x,
            'y': to_y
        }

def find_mask_bounding_box(mask: MatLike) -> MaskBoundingBox:
    conts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bb_dim = [[], []]

    for c in conts:
        x,y,w,h = cv2.boundingRect(c)

        if bb_dim[0].__len__() == 0:
            bb_dim[0] = [x, y]
        else:
            bb_dim[0][0] = min(bb_dim[0][0], x)
            bb_dim[0][1] = min(bb_dim[0][1], y)
        if bb_dim[1].__len__() == 0:
            bb_dim[1] = [x, y]
        else:
            bb_dim[1][0] = max(bb_dim[1][0], x+w)
            bb_dim[1][1] = max(bb_dim[1][1], y+h)

    return MaskBoundingBox(bb_dim[0][0], bb_dim[0][1], bb_dim[1][0], bb_dim[1][1])

def bgr_to_hsv(img_bgr: MatLike) -> MatLike:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
