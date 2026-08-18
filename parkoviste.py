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

# Nové proměnné pro stavový automat a časovač
spz = ""
hledat_zelenou = False
zelena_led_sviti = False
konecny_cas = 0

while True:
    ret, image = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Modrá maska pro detekci modré LED
    mask1 = image_tools.extract_color_mask_min(hsv, 100, 140)
    white_pixels = cv2.countNonZero(mask1)

    if white_pixels > MIN_PIXELS:
        if not led_sviti:
            print(f"Modrá LED rozsvícena! Stahuji obrázek...")

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

                # Po vypsání SPZ aktivujeme hledání zeleného světla
                hledat_zelenou = True

            except Exception as e:
                print(f"Chyba při stahování nebo OCR: {e}")

        led_sviti = True
    else:
        led_sviti = False

    # 2. Hledání zeleného světla po načtení SPZ
    if hledat_zelenou:
        mask_green = image_tools.extract_color_mask_min(hsv, 35, 85)
        green_pixels = cv2.countNonZero(mask_green)

        if green_pixels > MIN_PIXELS:
            if not zelena_led_sviti:
                print("Zelená LED rozsvícena! Přidávám 15 sekund...")
                konecny_cas = time.time() + 15
                zelena_led_sviti = True
        else:
            zelena_led_sviti = False

    # 3. Kontrola odpočtu času do 0
    if konecny_cas > 0:
        zbyvajici_cas = konecny_cas - time.time()

        if zbyvajici_cas <= 0:
            print(f"Předchozí SPZ {spz} musí odjet z parkoviště!")
            # Resetujeme časovač a hledání zelené pro další cyklus
            konecny_cas = 0
            hledat_zelenou = False

    cv2.imshow("Puvodni", mask1)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
