import cv2
import numpy as np
from cv2.typing import MatLike


def extract_color_mask_min(img_hsv: MatLike, hue_min: int, hue_max: int, sat_min: int = 100, sat_max: int = 255, val_min: int = 20, val_max: int = 255) -> MatLike:

                    #H         S       V
    min_c = np.array([hue_min, sat_min, val_min])
    max_c = np.array([hue_max, sat_max, val_max])

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

    def get_size(self) -> tuple[int, int]:
        return (abs(self.from_pos['x'] - self.to_pos['x']), abs(self.from_pos['y'] - self.to_pos['y']))

def find_mask_bounding_box(mask: MatLike) -> MaskBoundingBox:
    conts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(conts) == 0:
        return MaskBoundingBox(0, 0, 0, 0)

    print(f"FINDING CONTS: {conts.__len__()}")

    bb_dim = [[10**9, 10**9], [-1, -1]]

    for c in conts:
        x,y,w,h = cv2.boundingRect(c)

        print(f"|-> x:{x}, y:{y}, w:{w}, h:{h}")


        bb_dim[0][0] = min(bb_dim[0][0], x)
        bb_dim[0][1] = min(bb_dim[0][1], y)

        bb_dim[1][0] = max(bb_dim[1][0], x+w)
        bb_dim[1][1] = max(bb_dim[1][1], y+h)

    return MaskBoundingBox(bb_dim[0][0], bb_dim[0][1], bb_dim[1][0], bb_dim[1][1])

def crop_to_bounding_box(img: MatLike, bb: MaskBoundingBox) -> MatLike:
    return img[bb.from_pos["y"]:bb.to_pos["y"], bb.from_pos["x"]:bb.to_pos["x"]]

def bgr_to_hsv(img_bgr: MatLike) -> MatLike:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def find_local_box_center(box_w: int, box_h: int) -> tuple[int, int]:
    return (int(box_w / 2), int(box_h / 2))
