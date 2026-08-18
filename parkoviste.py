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

spz = ""
ceka_na_zelenou = False
ceka_na_druhou_modrou = False
konecny_cas = 0

while True:
    ret, image = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_blue = image_tools.extract_color_mask_min(hsv, 100, 140)
    blue_pixels = cv2.countNonZero(mask_blue)

    if konecny_cas == 0:
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

                    # Odpočet zbývajících 10 s (celkem 15 s od otevření závory)
                    konecny_cas = time.time() + 10
                    ceka_na_druhou_modrou = False

            led_sviti = True
        else:
            led_sviti = False

        if ceka_na_zelenou:
            mask_green = image_tools.extract_color_mask_min(hsv, 35, 85)
            green_pixels = cv2.countNonZero(mask_green)

            if green_pixels > MIN_PIXELS:
                print("Zelená LED rozsvícena! Nyní čekám na opětovné rozsvícení modré LED...")
                ceka_na_zelenou = False
                ceka_na_druhou_modrou = True

    if konecny_cas > 0:
        zbyvajici_cas = konecny_cas - time.time()

        if zbyvajici_cas <= 0:
            print(f"Čas vypršel! Auto s SPZ {spz} musí odjet z parkoviště!")
            # Reset kompletního stavu pro další auto
            konecny_cas = 0
            spz = ""

    cv2.imshow("Puvodni", mask_blue)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
