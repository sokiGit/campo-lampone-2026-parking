from ast import While
from multiprocessing import BoundedSemaphore
import sys

import cv2
import numpy as np
import requests

import image_tools

CROP_OUT_PADDING = 10
CONTOUR_AREA_MIN = 100
CONTOUR_AREA_MAX = 400
CROP_CHANGE_THRESHOLD = 10

whole_img_crop = image_tools.MaskBoundingBox(
    from_x=0,
    from_y=0,
    to_x=0,
    to_y=0
)

if __name__ == '__main__':
    while cv2.waitKey(1) != ord('q'):
        resp = requests.get("http://roofson.lan/", timeout=30)

        img_arr = np.frombuffer(resp.content, np.uint8)

        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if img is None:
            sys.exit()

        img = img[CROP_OUT_PADDING:-CROP_OUT_PADDING,CROP_OUT_PADDING:-CROP_OUT_PADDING,:]
        img = img[:,:,:]

        #img_whites = image_tools.extract_color_mask_min(img, 0, 180, sat_min=0, val_min=120)
        img_whites = image_tools.extract_color_mask_min(image_tools.bgr_to_hsv(img), 0, 180, sat_min=0, sat_max=50, val_min=120)
        img_bb = image_tools.find_mask_bounding_box(img_whites)

        if abs(whole_img_crop.from_pos['x'] - img_bb.from_pos['x']) >= CROP_CHANGE_THRESHOLD or \
           abs(whole_img_crop.to_pos['x'] - img_bb.to_pos['x']) >= CROP_CHANGE_THRESHOLD or \
           abs(whole_img_crop.from_pos['y'] - img_bb.from_pos['y']) >= CROP_CHANGE_THRESHOLD or \
           abs(whole_img_crop.to_pos['y'] - img_bb.to_pos['y']) >= CROP_CHANGE_THRESHOLD:
            whole_img_crop.from_pos['x'] = img_bb.from_pos['x']
            whole_img_crop.to_pos['x'] = img_bb.to_pos['x']
            whole_img_crop.from_pos['y'] = img_bb.from_pos['y']
            whole_img_crop.to_pos['y'] = img_bb.to_pos['y']

        img = image_tools.crop_to_bounding_box(img, whole_img_crop)
        img_whites = image_tools.crop_to_bounding_box(img_whites, whole_img_crop)

        img_h, img_w = img.shape[:2]

        #cv2.imshow("img_whites", img_whites)
        #
        hsv = image_tools.bgr_to_hsv(img)

        mask_r_0 = image_tools.extract_color_mask_min(hsv, 0, 10, sat_min=60, sat_max=255, val_min=100, val_max=255)
        mask_r_1 = image_tools.extract_color_mask_min(hsv, 160, 179, sat_min=60, sat_max=255, val_min=100, val_max=255)

        #cv2.imshow("Hue", hsv[:,:,0])
        #cv2.imshow("Saturation", hsv[:,:,1])
        #cv2.imshow("Value", hsv[:,:,2])

        #eroded_r = cv2.erode(mask_r_1, np.ones((2,2), np.uint8))
        #eroded_r = mask_r_1
        mask_r = cv2.bitwise_or(mask_r_0, mask_r_1)

        #cv2.imshow("car_detect_mask_cutout", cv2.bitwise_and(img[:,:,0], cv2.bitwise_not(img_whites)))
        #cv2.imshow("car_detect_mask_b", cv2.bitwise_and(img[:,:,0], mask_r))
        #cv2.imshow("car_detect_mask_g", cv2.bitwise_and(img[:,:,1], mask_r))
        #cv2.imshow("car_detect_mask_r", cv2.bitwise_and(img[:,:,2], mask_r))

        #eroded_r_l1 = cv2.erode(mask_r, np.ones((3,3), np.uint8))
        #eroded_r_l2 = cv2.erode(eroded_r_l1, np.ones((3,3), np.uint8))
        #eroded_r_l3 = cv2.erode(eroded_r_l2, np.ones((2,2), np.uint8))
        #eroded_r_l4 = cv2.erode(eroded_r_l3, np.ones((2,2), np.uint8))

        conts, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow("mask_r_0", mask_r_0)
        #cv2.imshow("mask_r_1", mask_r_1)

        #mask_r = cv2.bitwise_or(mask_r_0, mask_r_1)
        #mask_r = mask_r_1

        _ = cv2.drawContours(img, conts, -1, (255, 255, 0), cv2.FILLED)

        for c in conts:
            perim = cv2.arcLength(c, True)
            epsilon = 0.07 * perim
            approx = cv2.approxPolyDP(c, epsilon, True)
            c_area = cv2.contourArea(c)

            if len(approx) == 3 and c_area >= CONTOUR_AREA_MIN and c_area <= CONTOUR_AREA_MAX:
                img_cont = np.zeros((img_h, img_w), np.uint8)

                tri = approx
                out = img.copy()
                _ = cv2.polylines(img, [tri], isClosed=True, color=(63, 255, 127), thickness=2)
                _ = cv2.drawContours(img_cont, [c], 0, (255,255,255), cv2.FILLED)

                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Draw the circle at (cx, cy)
                    _ = cv2.circle(img, (cx, cy), 4, (255, 128, 64), -1)

                print(cv2.contourArea(c))
            else:
                print(f"Inadequate contour: Area: {c_area}px; Points: {len(approx)}")
                tri = approx
                out = img.copy()
                _ = cv2.polylines(img, [tri], isClosed=True, color=(255, 0, 63), thickness=2)
                #_ = cv2.drawContours(img, [hull_cont], -1, (255, 0, 63), cv2.FILLED)

        cv2.imshow("img_cont", img)
        #cv2.imshow("full_view", hsv)
        #cv2.imshow("full_view_mask", mask_r)
        #cv2.imshow("eroded_r", eroded_r)
        #
