import datetime
import urllib.request
import cv2
import easyocr
import numpy as np

import image_tools

# Inicializace EasyOCR (při prvním spuštění stáhne potřebné modely)
# Pokud máš dedikovanou grafickou kartu NVIDIA, nastav gpu=True pro zrychlení
reader = easyocr.Reader(["en"], gpu=False)

cap = cv2.VideoCapture("http://gateson.lan/stream")

# Minimální počet bílých pixelů pro aktivaci (přizpůsob podle potřeby)
MIN_PIXELS = 5
CAPTURE_URL = "http://gateson.lan/capture?delay=5"

# Pomocná proměnná pro sledování předchozího stavu
led_sviti = False

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
        # Uloží se pouze při PRVNÍM detekování rozsvícení (přechod z False na True)
        if not led_sviti:
            print(f"Modrá LED rozsvícena! Stahuji obrázek...")

            # Vytvoření unikátního názvu souboru s časovým razítkem
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"

            try:
                urllib.request.urlretrieve(CAPTURE_URL, filename)
                print(f"Obrázek uložen jako: {filename}")

                # --- Čtení SPZ pomocí EasyOCR ---
                results = reader.readtext(filename)

                spz = ""
                for bbox, text, prob in results:
                    clean_text = text.strip().replace(" ", "").upper()
                    # Filtrování krátkých šumů
                    if len(clean_text) >= 4:
                        spz = clean_text
                        break

                print(f"Přečtená SPZ: {spz}")

            except Exception as e:
                print(f"Chyba při stahování nebo OCR: {e}")

        led_sviti = True
    else:
        led_sviti = False

    cv2.imshow("Puvodni", mask1)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
