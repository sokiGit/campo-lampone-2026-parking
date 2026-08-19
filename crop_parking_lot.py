import sys
from math import floor
import random
from time import time

import cv2
import requests

import car_detector
import data_server
import image_tools

def find_cell(data_grid: image_tools.Grid, x: int, y: int):
    return next(
        (cell for cell in data_grid["cells"] if cell["x"] == x and cell["y"] == y),
        None,
    )

TRACKING_OCCUPANCY_TIMEOUT_MS = 200
CAR_TRACKING_LENIENCY_PX = 20

if __name__ == '__main__':
    img = image_tools.fetch_roofson_image()
    if img is None:
        sys.exit()

    img = img[10:-10,10:-10,:]

    data_grid: image_tools.Grid = image_tools.build_grid(img, return_debug=True)
    bb_w, bb_h = data_grid["bb"].get_size()
    n_cols, n_rows = data_grid["size"]

    image_tools.debug_draw_grid(img, data_grid)
    if data_grid.get("contours") is None:
        print("No grid contours found!")
        sys.exit()

    cv2.drawContours(img, data_grid["contours"], -1, (0, 255, 0), 3)

    # start server

    data_server.start(port=8001)

    active_cars = {}

    while cv2.waitKey(1) != ord("q"):

        img = image_tools.fetch_roofson_image()
        if img is None:
            continue

        img = img[10:-10,10:-10,:]

        previous_occupancies = []

        for cell in data_grid["cells"]:
            cell["occupied"] = False

        img_copy = img.copy()

        image_tools.debug_draw_grid(img, data_grid)

        car_positions = car_detector.detect_car_positions(img_copy)

        for cell in data_grid["cells"]:
            cell["occupied"] = False
            cell["reserved"] = False

        active_target_cells = {
            car["target_cell"] for car in active_cars.values() if car.get("target_cell")
        }

        for cell in data_grid["cells"]:
            if (cell["x"], cell["y"]) in active_target_cells:
                cell["reserved"] = True

        for car_pos_data in car_positions:
            car_pos = car_pos_data["pos"]
            angle = car_pos_data["angle"]

            _ = cv2.circle(img, car_pos, 6, (127, 255, 63), -1)

            car_grid_p = [
                floor((car_pos[0] - data_grid["bb"].from_pos["x"]) / (bb_w / n_cols)),
                floor((car_pos[1] - data_grid["bb"].from_pos["y"]) / (bb_h / n_rows)),
            ]

            car_pos_data["grid_pos"] = car_grid_p

            car_grid_cell = find_cell(data_grid, car_grid_p[0], car_grid_p[1])
            if car_grid_cell is not None:
                _ = cv2.circle(img, car_grid_cell["box_center"], 4, (255, 127, 63), -1)

                car_grid_cell["occupied"] = True
            else:
                car_grid_cell = None

            is_tracked = False

            for car_id, car_data in active_cars.items():
                dist = ((car_data["pos_px"][0] - car_pos[0])**2 + (car_data["pos_px"][1] - car_pos[1])**2)**0.5

                if dist <= CAR_TRACKING_LENIENCY_PX:
                    car_data["last_seen"] = time()
                    car_data["pos_px"] = car_pos
                    car_data["pos_grid"] = car_grid_p
                    car_data["angle"] = angle

                    is_tracked = True
                    # print(f"Continuing tracking for car: {car_id}")
                    break

            if not is_tracked:
                if car_grid_p[0] > data_grid["size"][0] - 1 or car_grid_p[1] > data_grid["size"][1] - 1:
                    car_id = f"car_{random.randint(0, 1000):x}"
                    print(f"Tracking new car: {car_id}")

                    spz_str = None
                    time_span = None
                    parking_type = []

                    try:
                        spz_req = requests.get("http://campo5.lan:8080/get_last_spz")

                        if spz_req.status_code == 200:
                            spz_data = spz_req.json()
                            spz_str = spz_data.get("spz", None)
                            time_span = spz_data.get("cas", None)
                            parking_type = [spz_data.get("typ_mista", None)]

                            #required_features = spz_data.get("required_features", [])
                    except requests.exceptions.RequestException as e:
                        print(f"Failed to fetch SPZ data using fallback\nError: {e}")

                    target_cell = car_detector.find_suitable_car_parking_spot(data_grid, parking_type)

                    active_cars[car_id] = {
                        "pos_px": car_pos,
                        "pos_grid": car_grid_p,
                        "angle": angle,
                        "last_seen": time(),
                        "spz": "debug_spz",#spz_str, #"debug_spz" # (for testing)
                        "target_cell": target_cell
                    }

                    if target_cell:
                        for cell in data_grid["cells"]:
                            if cell["x"] == target_cell[0] and cell["y"] == target_cell[1]:
                                cell["reserved"] = True
                                break

                else:
                    print("[WARN] New car is already within the grid!")
                    continue

            print(f"Car Grid Pos: {car_grid_p[0]}; {car_grid_p[1]}")

        current_time = time()
        timeout_sec = TRACKING_OCCUPANCY_TIMEOUT_MS / 1000.0

        for car_id in list(active_cars.keys()):
            if current_time - active_cars[car_id]["last_seen"] > timeout_sec:
                print(f"Lost track of {car_id}, removing.")
                del active_cars[car_id]

        if active_cars.__len__() == 0:
            print("No active cars.")
        else:
            print("Car tracking:")
            for car_id, car_data in active_cars.items():
                print(f"  {car_id}({car_data.get('spz') if car_data.get('spz') is not None else 'N/A'}): pos_px={car_data['pos_px']}, pos_grid={car_data['pos_grid']}, last_seen={car_data['last_seen']}")
                car_pos = car_data["pos_px"]
                spz_txt_pos = (car_pos[0] + 10, car_pos[1] + 5)
                car_pos_grid = car_data["pos_grid"]

                _ = cv2.putText(img, f"SPZ: {car_data.get('spz') if car_data.get('spz') is not None else 'N/A'}", spz_txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 255, 63), 1)
                angle = car_data.get('angle', -999)
                ang_txt_pos = (car_pos[0], car_pos[1] - 10)
                _ = cv2.putText(img, f"{angle:.0f} deg", ang_txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 255, 63), 1)
                _ = cv2.circle(img, car_pos, 3, (255, 127, 255), -1)
                target_cell = car_data.get('target_cell')
                if target_cell is not None:
                    tc_x, tc_y = target_cell
                    target_cell_pos = None
                    for cell in data_grid["cells"]:
                        if cell["x"] == tc_x and cell["y"] == tc_y:
                            target_cell_pos = cell["box_center"]
                            break
                    if target_cell_pos is not None:
                        _ = cv2.arrowedLine(img, car_pos, target_cell_pos, (127, 255, 63), 1)

        data_server.publish(
            data_grid["cells"],
            car_positions,
            active_cars
        )

        cv2.imshow("OUT", img)
