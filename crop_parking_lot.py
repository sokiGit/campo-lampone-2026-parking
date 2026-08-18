import sys
from math import floor
import random
from time import time

import cv2

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

    data_grid: image_tools.Grid = image_tools.build_grid(img, return_debug=True)
    bb_w, bb_h = data_grid["bb"].get_size()
    n_cols, n_rows = data_grid["size"]

    image_tools.debug_draw_grid(img, data_grid)
    if data_grid.get("contours") is None:
        print("No grid contours found!")
        sys.exit()

    cv2.drawContours(img, data_grid["contours"], -1, (0, 255, 0), 3)

    # start server

    data_server.start(port=8000)

    active_cars = {
        "ABC123": {
            "pos_px": (0, 0),
            "pos_grid": (0, 0),
            "last_seen": time(),
        }
    }

    while cv2.waitKey(1) != ord("q"):

        img = image_tools.fetch_roofson_image()
        if img is None:
            continue

        previous_occupancies = []

        for cell in data_grid["cells"]:
            cell["occupied"] = False

        img_copy = img.copy()

        image_tools.debug_draw_grid(img, data_grid)

        car_positions = car_detector.detect_car_positions(img_copy)

        for car_pos_data in car_positions:
            car_pos = car_pos_data["pos"]
            angle = car_pos_data["angle"]

            _ = cv2.circle(img, car_pos, 6, (127, 255, 63), -1)

            txt_pos = (car_pos[0], car_pos[1] - 10)

            _ = cv2.putText(img, f"{angle:.0f} deg", txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 255, 63), 1)

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
                # A more robust Euclidean distance check
                dist = ((car_data["pos_px"][0] - car_pos[0])**2 + (car_data["pos_px"][1] - car_pos[1])**2)**0.5

                if dist <= CAR_TRACKING_LENIENCY_PX:
                    car_data["last_seen"] = time()
                    car_data["pos_px"] = car_pos  # Keep the actual exact coordinate!
                    car_data["pos_grid"] = car_grid_p # Update the grid label
                    is_tracked = True
                    # print(f"Continuing tracking for car: {car_id}")
                    break

            if not is_tracked:
                car_id = f"car_{random.randint(0, 1000):x}"
                print(f"Tracking new car: {car_id}")

                active_cars[car_id] = {
                    "pos_px": car_pos,
                    "pos_grid": car_grid_p,
                    "last_seen": time(),
                }

            print(f"Car Grid Pos: {car_grid_p[0]}; {car_grid_p[1]}")

        current_time = time()
        timeout_sec = TRACKING_OCCUPANCY_TIMEOUT_MS / 1000.0

        for car_id in list(active_cars.keys()):
            if current_time - active_cars[car_id]["last_seen"] > timeout_sec:
                # print(f"Lost track of {car_id}, removing.")
                del active_cars[car_id]

        print("New car tracking:")
        for car_id, car_data in active_cars.items():
            print(f"  {car_id}: pos_px={car_data['pos_px']}, pos_grid={car_data['pos_grid']}, last_seen={car_data['last_seen']}")

        data_server.publish(data_grid["cells"], car_positions)

        cv2.imshow("OUT", img)
