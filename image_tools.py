from typing import TypedDict

import cv2
from cv2.mat_wrapper import Mat
import numpy as np
from cv2.typing import MatLike
import requests
from typing import Any, NotRequired, TypedDict

def fetch_roofson_image(vertical_crop_distance_px: int = 450) -> MatLike | None:
    img = None
    try:
        resp = requests.get("http://roofson.lan/", timeout=30)
        img_arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Failed to fetch roofson image: {e}")
    if img is None:
        return None
    return img[:vertical_crop_distance_px, :, :]   # consistent crop for grid + live frames

class _Pos(TypedDict):
    x: int
    y: int

class MaskBoundingBox:
    from_pos: _Pos
    to_pos: _Pos

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

class Cell(TypedDict):
    x: int
    y: int
    box_center: tuple[int, int]
    type: str
    occupied: bool
    reserved: bool
    features: list[str]


class Grid(TypedDict):
    cells: list[Cell]
    bb: MaskBoundingBox
    size: tuple[int, int]
    contours: NotRequired[Any]   # cv2 contour typing is awkward; Any is pragmatic
    mask: NotRequired[Any]


def build_grid(img: MatLike, return_debug: bool = False) -> Grid:
    GRID_SIZE = [5, 5]
    ROAD_COLS = [0, 2, 4]
    ROAD_ROWS = [0, 4]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_r_0 = extract_color_mask_min(hsv, 0, 10)
    hsv_r_1 = extract_color_mask_min(hsv, 170, 180)
    hsv_r = cv2.bitwise_or(hsv_r_0, hsv_r_1)

    conts, _ = cv2.findContours(hsv_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    grid_bb = find_mask_bounding_box(hsv_r)
    bb_w = grid_bb.to_pos['x'] - grid_bb.from_pos['x']
    bb_h = grid_bb.to_pos['y'] - grid_bb.from_pos['y']

    img_copy = img.copy()

    cells: list[Cell] = []
    for box_x in range(GRID_SIZE[0]):
        for box_y in range(GRID_SIZE[1]):
            box_x_px = [
                int(box_x * (bb_w / GRID_SIZE[0])),
                int((box_x + 1) * (bb_w / GRID_SIZE[0])),
            ]
            box_y_px = [
                int(box_y * (bb_h / GRID_SIZE[1])),
                int((box_y + 1) * (bb_h / GRID_SIZE[1])),
            ]

            box_center = find_local_box_center(
                box_x_px[1] - box_x_px[0],
                box_y_px[1] - box_y_px[0],
            )

            box_center_pos_total = (
                grid_bb.from_pos['x'] + box_x_px[0] + box_center[0],
                grid_bb.from_pos['y'] + box_y_px[0] + box_center[1],
            )

            if box_y == ROAD_ROWS[0] or box_y == ROAD_ROWS[-1]:
                # edge case (top/bottom roads)
                cell_type = "parking" if box_x == ROAD_COLS[0] or box_x == ROAD_COLS[-1] else "road"
            else:
                cell_type = "parking" if box_x in ROAD_COLS and box_y not in ROAD_ROWS else "road"


            # features
            cropped_cell_img = img[box_y_px[0] + grid_bb.from_pos['y']:box_y_px[1] + grid_bb.from_pos['y'], box_x_px[0] + grid_bb.from_pos['x']:box_x_px[1] + grid_bb.from_pos['x'], :]

            cropped_hsv = cv2.cvtColor(cropped_cell_img, cv2.COLOR_BGR2HSV)

            green_mask = cv2.erode(extract_color_mask_min(cropped_hsv, 60, 95, sat_min=127, val_min=127), np.ones((6, 6), np.uint8))
            blue_mask = cv2.erode(extract_color_mask_min(cropped_hsv, 100, 128, sat_min=127, val_min=127), np.ones((6, 6), np.uint8))

            #cv2.imshow(f'green_mask {box_x},{box_y}', green_mask)
            #cv2.imshow(f'blue_mask {box_x},{box_y}', blue_mask)

            contours_g, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_b, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            features = []

            if contours_g.__len__() > 0:
                features.append("electric_charger")
            if contours_b.__len__() > 0:
                features.append("disabled_only")

            _ = cv2.drawContours(cropped_cell_img, contours_g, -1, (0, 255, 0), 2)
            _ = cv2.drawContours(cropped_cell_img, contours_b, -1, (255, 0, 0), 2)

            #cv2.imshow(f'cropped_cell_img {box_x},{box_y}', cropped_cell_img)

            contours_g, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_g:
                _ = cv2.drawContours(img_copy, contours_g, -1, (255, 0, 255), 2)

            contours_b, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_b:
                _ = cv2.drawContours(img_copy, contours_b, -1, (255, 255, 0), 2)

            green_mask = cv2.dilate(green_mask, np.ones((2, 2), np.uint8))
            blue_mask = cv2.dilate(blue_mask, np.ones((2, 2), np.uint8))

            _ = cv2.rectangle(img_copy, (box_x_px[0] + grid_bb.from_pos['x'], box_y_px[0] + grid_bb.from_pos['y']), (box_x_px[1] + grid_bb.from_pos['x'], box_y_px[1] + grid_bb.from_pos['y']), (127, 63, 63), 1)

            cells.append({
                "x": box_x,
                "y": box_y,
                "box_center": box_center_pos_total,
                "type": cell_type,
                "occupied": False,
                "reserved": False,
                "features": features,
            })

    cv2.imshow("build_grid", img_copy)

    result: Grid = {
        "cells": cells,
        "bb": grid_bb,
        "size": (GRID_SIZE[0], GRID_SIZE[1]),   # explicit tuple[int, int]
    }

    if return_debug:
        # direct key assignment on an annotated TypedDict works fine:
        result["contours"] = conts
        result["mask"] = hsv_r

    return result

def debug_draw_grid(img: MatLike, grid: Grid):
    origin = (grid["bb"].from_pos['x'], grid["bb"].from_pos['y'])
    bb_w = grid["bb"].to_pos['x'] - grid["bb"].from_pos['x']
    bb_h = grid["bb"].to_pos['y'] - grid["bb"].from_pos['y']
    n_cols, n_rows = grid["size"]

    cv2.rectangle(
        img,
        origin,
        (grid["bb"].to_pos['x'], grid["bb"].to_pos['y']),
        (0, 127, 0), 2,
    )

    for cell in grid["cells"]:
        x0 = origin[0] + int(cell["x"] * (bb_w / n_cols))
        y0 = origin[1] + int(cell["y"] * (bb_h / n_rows))
        x1 = origin[0] + int((cell["x"] + 1) * (bb_w / n_cols))
        y1 = origin[1] + int((cell["y"] + 1) * (bb_h / n_rows))

        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 1)
        if cell["type"] == "parking":
            color = (0, 255, 0) if cell["reserved"] else (0, 127, 0)
            cv2.circle(img, cell["box_center"], 3, color, -1)  # yellow dot
        else:
            cv2.circle(img, cell["box_center"], 3, (0, 255, 255), -1)  # yellow dot
        if cell["features"] and cell["features"].__len__() > 0:
            txt = ""
            if "electric_charger" in cell["features"]:
                txt += "E"
            if "disabled_only" in cell["features"]:
                txt += "D"
            _ = cv2.putText(img, txt, (cell["box_center"][0], cell["box_center"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def extract_color_mask_min(img_hsv: MatLike, hue_min: int, hue_max: int, sat_min: int = 100, sat_max: int = 255, val_min: int = 20, val_max: int = 255) -> MatLike:

                    #H         S       V
    min_c = np.array([hue_min, sat_min, val_min])
    max_c = np.array([hue_max, sat_max, val_max])

    mask = cv2.inRange(img_hsv, min_c, max_c)
    return mask

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
