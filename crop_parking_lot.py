import json
import sys

import cv2
import numpy as np
import requests
from cv2.typing import MatLike

import image_tools

COLOR_THR = 150
ROAD_COLS = [0, 2, 4]
ROAD_ROWS = [0, 4]
GRID_SIZE = [5, 5]

def std_thresh(img_in: MatLike) -> MatLike:
    _, out = cv2.threshold(img_in, COLOR_THR, 255, cv2.THRESH_BINARY)
    return out

car_pos = [255, 255]

car_grid_p =

if __name__ == '__main__':
    resp = requests.get("http://roofson.lan/", timeout=30)
    img_arr = np.frombuffer(resp.content, np.uint8)

    # BGR
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if img is None:
        sys.exit()

    # Crop
    img = img[:450,:,:]

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV,)

    min_r_0 = np.array([0, 100, 20])
    max_r_0 = np.array([10, 255, 255])

    min_r_1 = np.array([170, 100, 20])
    max_r_1 = np.array([180, 255, 255])

    hsv_r_0 = image_tools.extract_color_mask_min(hsv, 0, 10)
    hsv_r_1 = image_tools.extract_color_mask_min(hsv, 170, 180)

    hsv_r = cv2.bitwise_or(hsv_r_0, hsv_r_1)

    conts, hierarchy = cv2.findContours(hsv_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    total_h, total_w = hsv.shape[:2]

    bbs = []

    bb_dim = [[], []]

    bb_img = np.zeros((total_h, total_w, 3))

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

    bb = image_tools.find_mask_bounding_box(hsv_r)

    bb_w = abs(bb.from_pos['x'] - bb.to_pos['x'])
    bb_h = abs(bb.from_pos['y'] - bb.to_pos['y'])

    data_grid = {
        "cells": []
    }

    for box_x in range(GRID_SIZE[0]):
        for box_y in range(GRID_SIZE[1]):
            box_x_px = [
                int(box_x * (bb_w / GRID_SIZE[0])),         #FROM
                int((box_x + 1) * (bb_w / GRID_SIZE[0]))    #TO
            ]
            box_y_px = [
                int(box_y * (bb_h / GRID_SIZE[1])),         #FROM
                int((box_y + 1) * (bb_h / GRID_SIZE[1]))    #TO
            ]
            cv2.rectangle(img, (bb_dim[0][0] + box_x_px[0], bb_dim[0][1] + box_y_px[0]), (bb_dim[0][0] + box_x_px[1], bb_dim[0][1] + box_y_px[1]), (255,0,0), 2)

            box_center = image_tools.find_local_box_center(
                box_x_px[1] - box_x_px[0],
                box_y_px[1] - box_y_px[0]
            )


            print(f"W: {box_x_px[1] - box_x_px[0]}; H: {box_y_px[1] - box_y_px[0]}")
            print(f"Box Center: {box_center[0]}; {box_center[1]}")

            box_center_pos_total = (
                bb_dim[0][0] + box_x_px[0] + box_center[0],
                bb_dim[0][1] + box_y_px[0] + box_center[1]
            )

            cell_type = "parking" if box_x in ROAD_COLS and box_y not in ROAD_ROWS else "road"
            features = [] # disabled_only, charging
            occupied = False

            data_grid["cells"].append({
                "x": box_x,
                "y": box_y,
                "box_center": box_center_pos_total,
                "type": cell_type,
                "occupied": occupied,
                "features": features
            })

            _ = cv2.circle(
                img,
                (
                    box_center_pos_total[0],
                    box_center_pos_total[1]
                ),
                4,
                (255, 255, 0),
                -1
            )

        print("drawing cont rect")

    _ = cv2.rectangle(img,(bb_dim[0][0],bb_dim[0][1]),(bb_dim[1][0], bb_dim[1][1]),(0,255,0),2)

    _ = cv2.circle(
        img,
        (car_pos[0], car_pos[1]),
        6,
        (127, 255, 63),
        -1
    )
    cv2.imshow("OUT", img)

    tr = None

    print(json.dumps(data_grid))

    # Charger:  Green
    # Disabled: Blue

    _ = cv2.drawContours(img, conts, -1, (0, 255, 0), 3)




    #cv2.imshow("OUT", img)

    #img_0 = std_thresh(img[:,:,0])
    #img_1 = std_thresh(img[:,:,1])
    #img_bg = cv2.bitwise_not(cv2.bitwise_or(img_0, img_1))
    #img_r = std_thresh(img[:,:,2])

    # cv2.imshow("OUT_0", img_bg)

    #img_bw = cv2.bitwise_and(img_r, img_bg)

    #img_r_thr = std_thresh(img_1)

    #cv2.imshow("OUT", img_bw)

    #cv2.imshow("Output", img_r)

    cv2.waitKey(0)
