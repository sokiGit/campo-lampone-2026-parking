import datetime
import threading
import time
import urllib.request
import cv2
import easyocr
import numpy as np
import requests

import gate_server
import image_tools

# --- CONFIGURATION ---
STREAM_URL = "http://gateson.lan/stream"
CAPTURE_URL = "http://gateson.lan/capture?delay=5"
GATE_OPEN_URL = "http://gateson.lan/gate_in?open"
GATE_CLOSE_URL = "http://gateson.lan/gate_in?close"

# Higher pixel threshold + morphological filtering prevents background noise triggers
MIN_PIXELS = 400
DEBOUNCE_COOLDOWN = 0.8  # Seconds to ignore repeated LED triggers


# --- THREADED CAMERA STREAM (Prevents Video Buffer Lag) ---
class LiveVideoCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()


# --- HELPER FUNCTIONS ---
def remove_mask_noise(mask):
    """Applies morphological opening to eliminate scattered single-pixel noise."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# Initialize components
reader = easyocr.Reader(["en"], gpu=False)
cap = LiveVideoCapture(STREAM_URL)
gate_server.start()

# State Machine Variables
faze = 1
spz = ""
cas = 0
pocet_zelenych_misto = 0
typ_mista = ""

# Button tracking & debouncing
modra_sviti_predtem = False
zelena_sviti_predtem = False
last_blue_press = 0.0
last_green_press = 0.0

# Async OCR flags
ocr_in_progress = False


def process_ocr_async():
    """Runs hi-res capture and EasyOCR in a background thread to keep video smooth."""
    global spz, faze, ocr_in_progress
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"

    try:
        print("1. Stahuji obrázek z kamery...")
        urllib.request.urlretrieve(CAPTURE_URL, filename)
        print(f"Obrázek uložen: {filename}. Spouštím OCR...")

        results = reader.readtext(filename)
        detected_spz = ""
        for bbox, text, prob in results:
            clean_text = text.strip().replace(" ", "").upper()
            if len(clean_text) >= 4:
                detected_spz = clean_text
                break

        spz = detected_spz
        print(f"Přečtená SPZ: '{spz}'")

        # Publish initial SPZ detection to server if needed
        if hasattr(gate_server, "publish_spz"):
            gate_server.publish_spz(spz)
        else:
            gate_server.publish(spz)

        # Move to Phase 2 (Time input)
        faze = 2
        print("Nyní zadávejte čas pomocí zelené LED (1x = +15s)...")

    except Exception as e:
        print(f"Chyba při stahování nebo OCR: {e}")
        faze = 1  # Reset to start on error

    finally:
        ocr_in_progress = False


def process_gate_and_commit(spz, cas, typ_mista):
    """Handles opening gate, waiting, closing gate, publishing complete data, and resetting."""
    global faze, spz_var, cas_var, pocet_zelenych_misto

    print(f"Otevírám závoru pro SPZ: {spz} ({typ_mista}, čas: {cas}s)")
    try:
        urllib.request.urlopen(GATE_OPEN_URL)
        time.sleep(5)
        print("Zavírám závoru.")
        urllib.request.urlopen(GATE_CLOSE_URL)
    except Exception as e:
        print(f"Chyba při ovládání závory: {e}")

    # Commit/Publish all gathered data to gate_server
    # Modify these method calls according to your gate_server implementation!
    if hasattr(gate_server, "publish_entry_data"):
        gate_server.publish_entry_data(spz=spz, parking_time=cas, spot_type=typ_mista)
    else:
        gate_server.publish(f"ENTRY:{spz},{cas},{typ_mista}")

    print("Data uložena/odeslána. Připraveno pro další auto (Fáze 1).\n")


# --- MAIN LOOP ---
while True:
    ret, image = cap.read()
    if not ret or image is None:
        time.sleep(0.01)
        continue

    # Color extraction and noise filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_blue = remove_mask_noise(image_tools.extract_color_mask_min(hsv, 100, 140))
    blue_pixels = cv2.countNonZero(mask_blue)

    mask_green = remove_mask_noise(image_tools.extract_color_mask_min(hsv, 35, 85))
    green_pixels = cv2.countNonZero(mask_green)

    now = time.time()

    # Blue LED edge detection with debouncing
    modra_stisknuta = False
    if blue_pixels > MIN_PIXELS:
        if not modra_sviti_predtem and (now - last_blue_press > DEBOUNCE_COOLDOWN):
            modra_stisknuta = True
            last_blue_press = now
        modra_sviti_predtem = True
    else:
        modra_sviti_predtem = False

    # Green LED edge detection with debouncing
    zelena_stisknuta = False
    if green_pixels > MIN_PIXELS:
        if not zelena_sviti_predtem and (now - last_green_press > DEBOUNCE_COOLDOWN):
            zelena_stisknuta = True
            last_green_press = now
        zelena_sviti_predtem = True
    else:
        zelena_sviti_predtem = False

    # --- STATE MACHINE ---

    # FÁZE 1: Čekání na příjezd auta (Modrá LED)
    if faze == 1:
        if modra_stisknuta and not ocr_in_progress:
            print("1. Modrá LED detekována (Auto přijelo). Spouštím OCR...")
            ocr_in_progress = True
            cas = 0
            pocet_zelenych_misto = 0
            faze = 10  # Interim phase while OCR runs in background
            threading.Thread(target=process_ocr_async, daemon=True).start()

    # FÁZE 10: Zpracovává se OCR v pozadí (Ignoruje stisky tlačítka)
    elif faze == 10:
        pass

    # FÁZE 2: Zadávání času (Zelená = +15s, Modrá = Potvrdit)
    elif faze == 2:
        if zelena_stisknuta:
            cas += 15
            print(f"Přidáno +15s! Celkový čas: {cas} s")

        elif modra_stisknuta:
            print(f"2. Modrá LED (Čas potvrzen na {cas} s). Vyberte typ místa zelenou LED...")
            pocet_zelenych_misto = 0
            faze = 4

    # FÁZE 4: Výběr místa (Zelená = přepínání, Modrá = Potvrdit & Otevřít závoru)
    elif faze == 4:
        if zelena_stisknuta:
            pocet_zelenych_misto += 1
            print(f"Zelená LED pro místo stisknuta ({pocet_zelenych_misto}x)")

        elif modra_stisknuta:
            if pocet_zelenych_misto == 1:
                typ_mista = "elektro"
            elif pocet_zelenych_misto == 2:
                typ_mista = "invalida"
            else:
                typ_mista = "Normální parkovací místo"

            print(f"3. Modrá LED (Místo potvrzeno: {typ_mista})")

            # Execute gate opening and data commit in a separate thread so it doesn't block
            threading.Thread(
                target=process_gate_and_commit,
                args=(spz, cas, typ_mista),
                daemon=True,
            ).start()

            # Immediately reset to Phase 1 so the next car can enter right away!
            faze = 1

    # Display video frame for debug
    cv2.imshow("Puvodni", mask_blue)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
