import cv2
import numpy as np

import image_tools

cap = cv2.VideoCapture("http://gateson.lan/stream")

while True:
    ret, image = cap.read()
    if not ret:
        break

    # 1. Převod obrázku z BGR do HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. Definoování rozsahů pro červenou barvu v HSV
    # Červená barva v HSV přetéká přes hranici 0/180°, proto potřebujeme dva rozsahy
    # Hodnota S (saturace) nastavená na min. 100 spolehlivě odfiltruje bílou (bílá má S blízko 0)
    lower_red1 = np.array([40, 100, 100])
    upper_red1 = np.array([60, 255, 255])

    # 3. Vytvoření masek
    mask1 = image_tools.extract_color_mask_min(hsv, 100, 140)

    # 4. Spojení obou masek
    # Výsledná maska obsahuje 255 (bílou) na místech červené a 0 (černou) jinde


    # Zobrazení
    cv2.imshow("Původní obraz", mask1)
    #cv2.imshow("Červená jako bílá", red_as_white)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
