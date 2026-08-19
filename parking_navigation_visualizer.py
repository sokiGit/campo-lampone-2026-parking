import cv2
import numpy as np
import requests


def visualize(spz: str):
    url_params = {'spz': spz}
    base_url = "http://campo5.lan:8000/get_my_data"

    while cv2.waitKey(1) != ord('q'):
        try:
            response = requests.get(base_url, params=url_params)

            data = response.json()
            print(data)

            cells = data['cells']

            max_x = max(cell['box_center'][0] for cell in cells)
            max_y = max(cell['box_center'][1] for cell in cells)

            img = np.zeros((int(max_y) + 100, int(max_x) + 100, 3), dtype=np.uint8)

            for cell in cells:
                x = int(cell['box_center'][0])
                y = int(cell['box_center'][1])
                color = (127, 127, 127) if cell['type'] == 'road' else (0, 127, 255) if cell['reserved'] else (63, 127, 255)
                if cell['occupied']:
                    color = (0, 0, 255)
                    s = 5  # half-size of the cross (tweak)
                    _ = cv2.line(img, (x - s, y - s), (x + s, y + s), color, 2, cv2.LINE_AA)
                    _ = cv2.line(img, (x + s, y - s), (x - s, y + s), color, 2, cv2.LINE_AA)

                _ = cv2.circle(img, (x, y), 4, color, -1)

            my_data = data['car_tracking_data']
            pos_px = my_data['pos_px']
            angle = np.deg2rad(my_data['angle'])
            _ = cv2.putText(img, f"{int(pos_px[0])}, {int(pos_px[1])}", (int(pos_px[0] + 10), int(pos_px[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2, cv2.LINE_AA)
            _ = cv2.putText(img, f"{angle}°", (int(pos_px[0] + 10), int(pos_px[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2, cv2.LINE_AA)
            _ = cv2.arrowedLine(img, (int(pos_px[0]), int(pos_px[1])), (int(pos_px[0] + 25 * np.cos(angle)), int(pos_px[1] + 25 * np.sin(angle))), (0, 255, 0), 1)

            _ = cv2.circle(img, (int(pos_px[0]), int(pos_px[1])), 4, (0, 255, 0), -1)

            target_cell = my_data['target_cell']
            target_x = -999
            target_y = -999

            for cell in cells:
                if cell['x'] == target_cell[0] and cell['y'] == target_cell[1]:
                    target_x = cell['box_center'][0]
                    target_y = cell['box_center'][1]

            _ = cv2.circle(img, (int(target_x), int(target_y)), 2, (0, 255, 255), -1)

            _ = cv2.arrowedLine(img, (int(pos_px[0]), int(pos_px[1])), (int(target_x), int(target_y)), (0, 127, 127), 1)

            cv2.imshow('img', img)
        except Exception as e:
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            _ = cv2.putText(img, f'Error: {e}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow('img', img)
    cv2.destroyAllWindows()
