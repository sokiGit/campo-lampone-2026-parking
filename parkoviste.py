import datetime
import time
import urllib.request
import cv2
import easyocr
import numpy as np
import requests

import image_tools

reader = easyocr.Reader(["en"], gpu=False)

cap = cv2.VideoCapture("http://gateson.lan/stream")

MIN_PIXELS = 100
CAPTURE_URL = "http://gateson.lan/capture?delay=5"

led_sviti = False
zelena_sviti_predtem = False

spz = ""
ceka_na_zelenou = False
ceka_na_druhou_modrou = False
odpocitavani_bezi = False

cas = 0
posledni_sekunda = 0

while True:
    ret, image = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_blue = image_tools.extract_color_mask_min(hsv, 100, 140)
    blue_pixels = cv2.countNonZero(mask_blue)

    if not odpocitavani_bezi:
        if blue_pixels > MIN_PIXELS:
            if not led_sviti:
                if not ceka_na_zelenou and not ceka_na_druhou_modrou:
                    print("Modrá LED rozsvícena! Stahuji obrázek...")

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{timestamp}.jpg"

                    try:
                        urllib.request.urlretrieve(CAPTURE_URL, filename)
                        print(f"Obrázek uložen jako: {filename}")

                        results = reader.readtext(filename)

                        spz = ""
                        for bbox, text, prob in results:
                            clean_text = text.strip().replace(" ", "").upper()
                            if len(clean_text) >= 4:
                                spz = clean_text
                                break

                        print(f"Přečtená SPZ: {spz}")
                        ceka_na_zelenou = True

                    except Exception as e:
                        print(f"Chyba při stahování nebo OCR: {e}")

                elif ceka_na_druhou_modrou:
                    print("Druhá modrá LED detekována! Ignoruji další barvy.")
                    print("Otevírám závoru.")
                    urllib.request.urlopen("http://gateson.lan/gate_in?open")

                    time.sleep(5)

                    print("Zavírám závoru.")
                    urllib.request.urlopen("http://gateson.lan/gate_in?close")

                    odpocitavani_bezi = True
                    posledni_sekunda = time.time()
                    ceka_na_druhou_modrou = False
                    print(f"Začíná odpočet! Zbývá: {cas} s")

            led_sviti = True
        else:
            led_sviti = False

        if ceka_na_zelenou or ceka_na_druhou_modrou:
            mask_green = image_tools.extract_color_mask_min(hsv, 35, 85)
            green_pixels = cv2.countNonZero(mask_green)

            if green_pixels > MIN_PIXELS:
                if not zelena_sviti_predtem:
                    cas += 15
                    print(f"Zelená rozsvícena! Proměnná cas: {cas} s")
                    zelena_sviti_predtem = True
                    ceka_na_zelenou = False
                    ceka_na_druhou_modrou = True
            else:
                zelena_sviti_predtem = False
                
    else:
        if time.time() - posledni_sekunda >= 1.0:
            cas -= 1
            posledni_sekunda = time.time()

            if cas <= 0:
                print(f"Čas vypršel! Auto s SPZ {spz} musí odjet z parkoviště!")
                cas = 0
                spz = ""
                odpocitavani_bezi = False

    cv2.imshow("Puvodni", mask_blue)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
