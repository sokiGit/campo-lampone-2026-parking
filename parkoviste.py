import datetime
import time
import urllib.request
import cv2
import easyocr
import numpy as np
import requests
import gate_server

import image_tools

reader = easyocr.Reader(["en"], gpu=False)

cap = cv2.VideoCapture("http://gateson.lan/stream")

MIN_PIXELS = 100
CAPTURE_URL = "http://gateson.lan/capture?delay=5"

modra_sviti_predtem = False
zelena_sviti_predtem = False

spz = ""
cas = 0
faze = 1
pocet_zelenych_misto = 0
posledni_sekunda = 0
typ_mista = ""

gate_server.start()

# Pomocná funkce pro vyprázdnění bufferu kamery
def vycistit_buffer_kamery(cap, pocet_snimku=15):
    for _ in range(pocet_snimku):
        cap.grab()

while True:
    ret, image = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_blue = image_tools.extract_color_mask_min(hsv, 100, 140)
    blue_pixels = cv2.countNonZero(mask_blue)

    mask_green = image_tools.extract_color_mask_min(hsv, 35, 85)
    green_pixels = cv2.countNonZero(mask_green)

    modra_stisknuta = False
    if blue_pixels > MIN_PIXELS:
        if not modra_sviti_predtem:
            modra_stisknuta = True
        modra_sviti_predtem = True
    else:
        modra_sviti_predtem = False

    zelena_stisknuta = False
    if green_pixels > MIN_PIXELS:
        if not zelena_sviti_predtem:
            zelena_stisknuta = True
        zelena_sviti_predtem = True
    else:
        zelena_sviti_predtem = False

    # faze 1
    if faze == 1:
        if modra_stisknuta:
            print("1. Modrá LED detekována (Auto přijelo). Stahuji obrázek...")
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
                        gate_server.publish(spz)
                        break

                print(f"Přečtená SPZ: {spz}")
                cas = 0
                faze = 2
                print("Nyní zadávejte čas pomocí zelené LED (1x = +15s)...")

            except Exception as e:
                print(f"Chyba při stahování nebo OCR: {e}")

            # Delay a vyčištění vyrovnávací paměti kamery po dlouhé operaci (OCR)
            time.sleep(1.0)
            vycistit_buffer_kamery(cap, 25)
            modra_sviti_predtem = False
            zelena_sviti_predtem = False

    # faze 2
    elif faze == 2:
        if zelena_stisknuta:
            cas += 15
            print(f"Přidáno +15s! Celkový čas: {cas} s")
            time.sleep(1.0)
            vycistit_buffer_kamery(cap, 15)
            zelena_sviti_predtem = False

        elif modra_stisknuta:
            print(f"2. Modrá LED (Čas potvrzen na {cas} s). Vyberte typ místa zelenou LED...")
            pocet_zelenych_misto = 0
            faze = 4
            time.sleep(1.0)
            vycistit_buffer_kamery(cap, 15)
            modra_sviti_predtem = False

    # faze 4
    elif faze == 4:
        if zelena_stisknuta:
            pocet_zelenych_misto += 1
            print(f"Zelená LED pro místo stisknuta ({pocet_zelenych_misto}x)")
            time.sleep(1.0)
            vycistit_buffer_kamery(cap, 15)
            zelena_sviti_predtem = False

        elif modra_stisknuta:
            if pocet_zelenych_misto == 1:
                typ_mista = "elektro"
            elif pocet_zelenych_misto == 2:
                typ_mista = "invalida"
            else:
                typ_mista = "Normální parkovací místo"

            print(f"3. Modrá LED (Místo potvrzeno): {typ_mista}")
            print("Otevírám závoru.")
            urllib.request.urlopen("http://gateson.lan/gate_in?open")

            time.sleep(5)

            print("Zavírám závoru.")
            urllib.request.urlopen("http://gateson.lan/gate_in?close")

            # Vyčištění starých snímků nastřádaných během 5s čekání u závory
            time.sleep(1.0)
            vycistit_buffer_kamery(cap, 25)

            posledni_sekunda = time.time()
            faze = 5
            print(f"Začíná odpočet času! Zbývá: {cas} s")
            modra_sviti_predtem = False
            zelena_sviti_predtem = False

    # faze 5
    elif faze == 5:
        if time.time() - posledni_sekunda >= 1.0:
            cas -= 1
            posledni_sekunda = time.time()

            if cas <= 0:
                print(f"Čas vypršel! Auto s SPZ {spz} ({typ_mista}) musí odjet z parkoviště!")
                cas = 0
                pocet_zelenych_misto = 0
                faze = 1
                time.sleep(1.0)
                vycistit_buffer_kamery(cap, 15)

    cv2.imshow("Puvodni", mask_blue)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
