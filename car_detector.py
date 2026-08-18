from ast import While
from multiprocessing import BoundedSemaphore
import re
import sys

import cv2
from cv2.typing import MatLike
import math
import numpy as np
import requests

import image_tools

CONTOUR_AREA_MIN = 100
CONTOUR_AREA_MAX = 400

def triangle_pointing_angle(pts):
    """pts: 3x2 array of the triangle vertices. Returns the arrow's
    pointing angle in degrees (0 = pointing right, CCW in math space;
    note image y is flipped)."""
    a, b, c = [tuple(p) for p in pts]

    sides = {
        "ab": math.hypot(b[0]-a[0], b[1]-a[1]),
        "bc": math.hypot(c[0]-b[0], c[1]-b[1]),
        "ca": math.hypot(a[0]-c[0], a[1]-c[1]),
    }
    shortest = min(sides, key=sides.get)

    if shortest == "ab":
        base_ex = (a, b); apex = c
    elif shortest == "bc":
        base_ex = (b, c); apex = a
    else:  # "ca"
        base_ex = (c, a); apex = b

    mid = ((base_ex[0][0] + base_ex[1][0]) / 2,
           (base_ex[0][1] + base_ex[1][1]) / 2)

    dx = apex[0] - mid[0]
    dy = apex[1] - mid[1]

    return math.degrees(math.atan2(dy, dx))

def detect_car_positions(img: MatLike):
    img_h, img_w = img.shape[:2]

    hsv = image_tools.bgr_to_hsv(img)

    mask_r_0 = image_tools.extract_color_mask_min(hsv, 0, 10, sat_min=60, sat_max=255, val_min=100, val_max=255)
    mask_r_1 = image_tools.extract_color_mask_min(hsv, 160, 179, sat_min=60, sat_max=255, val_min=100, val_max=255)
    mask_r = cv2.bitwise_or(mask_r_0, mask_r_1)

    # debug show mask
    cv2.imshow("mask_r_debug", mask_r)

    conts, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    car_positions = []

    for c in conts:
        perim = cv2.arcLength(c, True)
        epsilon = 0.07 * perim
        approx = cv2.approxPolyDP(c, epsilon, True)
        c_area = cv2.contourArea(c)

        if len(approx) == 3 and c_area >= CONTOUR_AREA_MIN and c_area <= CONTOUR_AREA_MAX:
            img_cont = np.zeros((img_h, img_w), np.uint8)

            tri = approx
            _ = cv2.polylines(img, [tri], isClosed=True, color=(63, 255, 127), thickness=2)
            _ = cv2.drawContours(img_cont, [c], 0, (255,255,255), cv2.FILLED)

            pts = tri.reshape(3, 2)
            angle = triangle_pointing_angle(pts)

            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                car_positions.append({"pos": (cx, cy), "angle": angle})
        else:
            continue

    return car_positions
