import datetime
import time
import urllib.request
import cv2
import easyocr
import numpy as np

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

    # 1. Kontrola barvy probíhá JEN pokud ještě neodpočítáváme čas po závoře
    if konecny_cas == 0:
        if blue_pixels > MIN_PIXELS:
            if not led_sviti:
                # FÁZE A: První modrá LED -> Stáhnutí obrázku a OCR
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

                # FÁZE C: Druhá modrá LED -> Ignorování dalších barev, závora a start časovače
                elif ceka_na_druhou_modrou:
                    print("Druhá modrá LED detekována! Ignoruji další barvy.")
                    print("Otevírám závoru...")
                    time.sleep(5)  # Závora otevřená 5 sekund
                    print("Zavírám závoru.")

                    # Odpočet zbývajících 10 s (celkem 15 s od otevření závory)
                    konecny_cas = time.time() + 10
                    ceka_na_druhou_modrou = False

            led_sviti = True
        else:
            led_sviti = False

        # FÁZE B: Čekání na zelenou LED
        if ceka_na_zelenou:
            mask_green = image_tools.extract_color_mask_min(hsv, 35, 85)
            green_pixels = cv2.countNonZero(mask_green)

            if green_pixels > MIN_PIXELS:
                print("Zelená LED rozsvícena! Nyní čekám na opětovné rozsvícení modré LED...")
                ceka_na_zelenou = False
                ceka_na_druhou_modrou = True

    # 2. Odpočet času do odjezdu (beží na pozadí neovlivněn obrazem)
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
