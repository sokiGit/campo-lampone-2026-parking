from colorsys import rgb_to_hls
import cv2
import numpy as np
import requests

response = requests.get(url="http://roofson.lan/", timeout=10)
image_array = np.frombuffer(response.content, np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
image = image[20:950, 15:575,]
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 150, 600, 420, 780
ROI2_X1, ROI2_Y1, ROI2_X2, ROI2_Y2 = 16, 600, 575, 960
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_red_1 = np.array([0, 60, 100])
upper_red_1 = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
lower_red_2 = np.array([160, 60, 100])
upper_red_2 = np.array([179, 255, 255])
mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
mask = cv2.bitwise_or(mask1, mask2)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
checkpoints = []
centers = []
location_jetson = []

for c in contours:

    if not (100 < cv2.contourArea(c) < 400):
        continue

    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    if not (ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2):
        continue

    checkpoints.append(c)
    centers.append((cx, cy))

print(centers[0])

for c in contours:

    # Aproximace kontury pro zjištění počtu vrcholů
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # Ověření, zda má tvar 3 vrcholy (trojúhelník)
    if len(approx) == 3:
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            if not (ROI2_X1 <= cx <= ROI2_X2 and ROI2_Y1 <= cy <= ROI2_Y2):
                continue


            location_jetson = (cx, cy)
print(location_jetson)

cv2.drawContours(image, checkpoints, -1, (0, 255, 0), 2)
for (cx, cy) in centers:
    cv2.circle(image, (cx, cy), 3, (255, 0, 0), -1)
    cv2.circle(image, location_jetson, 3, (255, 0, 0), -1)
cv2.rectangle(image, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 1)
cv2.rectangle(image, (ROI2_X1, ROI2_Y1), (ROI2_X2, ROI2_Y2), (0, 255, 0), 1)

tolerance = 10

# todo: zlepšit
target_center = centers[0]
print(target_center)

distance = np.sqrt(
        (location_jetson[0] - target_center[0]) ** 2
) if location_jetson.__len__() > 0 else -1
print(distance)
if distance <= tolerance:
        print("Payload spuštěn: Pozice odpovídá s požadovanou tolerancí.")
        # exit()

cv2.imshow("Output", image)
cv2.waitKey(0)
