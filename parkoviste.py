import cv2
import numpy as np

import image_tools

cap = cv2.VideoCapture("http://gateson.lan/stream")

# Minimální počet bílých pixelů pro aktivaci (přizpůsob podle potřeby)
MIN_PIXELS = 100

while True:
    ret, image = cap.read()
    if not ret:
        break

    # 1. Převod obrázku z BGR do HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. Vytvoření masek
    mask1 = image_tools.extract_color_mask_min(hsv, 100, 140)

    # Spočítání bílých pixelů v masce
    white_pixels = cv2.countNonZero(mask1)

    # Podmínka, která vrátí True při rozsvícení LED
    if white_pixels > MIN_PIXELS:
        led_sviti = True
        print(f"Modrá LED svítí! ({white_pixels} pixelů)")
    else:
        led_sviti = False

    cv2.imshow("Původní obraz", mask1)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
