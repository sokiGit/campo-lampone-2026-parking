import sys

import cv2
import numpy as np
import requests
from cv2.typing import MatLike

COLOR_THR = 150

def std_thresh(img_in: MatLike) -> MatLike:
    _, out = cv2.threshold(img_in, COLOR_THR, 255, cv2.THRESH_BINARY)
    return out

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

    hsv_r_0 = cv2.inRange(hsv, min_r_0, max_r_0)
    hsv_r_1 = cv2.inRange(hsv, min_r_1, max_r_1)

    hsv_r = cv2.bitwise_or(hsv_r_0, hsv_r_1)

    conts, hierarchy = cv2.findContours(hsv_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print(conts)

    total_h, total_w, _ = hsv.shape

    bbs = []

    bb_dim = [[], []]

    bb_img = np.zeros((total_h, total_w, 3))

    grid_size = [5, 5]

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

    bb_w = abs(bb_dim[0][0] - bb_dim[1][0])
    bb_h = abs(bb_dim[0][1] - bb_dim[1][1])

    for box_x in range(grid_size[0]):
        for box_y in range(grid_size[1]):
            box_x_px = [
                int(box_x * (bb_w / grid_size[0])),
                int((box_x + 1) * (bb_w / grid_size[0]))
            ]
            box_y_px = [
                int(box_y * (bb_h / grid_size[1])),
                int((box_y + 1) * (bb_h / grid_size[1]))
            ]
            cv2.rectangle(img, (bb_dim[0][0] + box_x_px[0], bb_dim[0][1] + box_y_px[0]), (bb_dim[0][0] + box_x_px[1], bb_dim[0][1] + box_y_px[1]), (255,0,0), 2)

        print("drawing cont rect")

    cv2.rectangle(img,(bb_dim[0][0],bb_dim[0][1]),(bb_dim[1][0], bb_dim[1][1]),(0,255,0),2)

    cv2.imshow("OUT", img)

    tr = None

    # Charger:  Green
    # Disabled: Blue

    cv2.drawContours(img, conts, -1, (0, 255, 0), 3)

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
