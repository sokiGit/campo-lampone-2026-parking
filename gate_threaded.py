import datetime
import json
import re
import threading
import time
import urllib.request
import cv2
import easyocr
import numpy as np

import gate_server
import image_tools

# --- CONFIGURATION ---
STREAM_URL = "http://gateson.lan/stream"
CAPTURE_URL = "http://gateson.lan/capture?delay=5"
GATE_OPEN_URL = "http://gateson.lan/gate_in?open"
GATE_CLOSE_URL = "http://gateson.lan/gate_in?close"

MIN_PIXELS = 150
DEBOUNCE_COOLDOWN = 0.6  # Seconds to ignore repeated LED triggers


# --- THREADED CAMERA STREAM ---
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
    """Applies a gentle morphological open to remove tiny isolated single-pixel noise."""
    kernel = np.ones((2, 2), np.uint8)
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

# Async flags & Locks
ocr_in_progress = False
phase_transition_lock = 0.0


def set_phase(new_phase, delay=1.5):
    """
    Safely transitions phases and locks input processing for 'delay' seconds.
    This allows the physical LEDs to settle without triggering false positives.
    """
    global faze, phase_transition_lock
    faze = new_phase
    phase_transition_lock = time.time() + delay


def process_ocr_async():
    """Runs image download and EasyOCR completely in the background."""
    global spz, ocr_in_progress
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"

        print("Stahuji snímek pro OCR na pozadí...")
        urllib.request.urlretrieve(CAPTURE_URL, filename)

        print("Spouštím OCR rozpoznávání...")
        results = reader.readtext(filename, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        detected_spz = ""
        best_prob = 0.0

        for bbox, text, prob in results:
            clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())
            if len(clean_text) >= 4 and prob > best_prob:
                detected_spz = clean_text
                best_prob = prob

        spz = detected_spz
        print(f"\n[OCR Dokončeno] Přečtená SPZ: '{spz}' (istota: {best_prob:.2f})")
        gate_server.publish(spz)

    except Exception as e:
        print(f"Chyba při OCR: {e}")

    finally:
        ocr_in_progress = False


def process_gate_and_commit(spz_val, cas_val, typ_mista_val):
    """Handles opening gate, waiting, closing gate, and committing data."""
    print(f"\nOtevírám závoru pro SPZ: {spz_val} ({typ_mista_val}, čas: {cas_val}s)")
    try:
        urllib.request.urlopen(GATE_OPEN_URL)
        time.sleep(10)
        print("Zavírám závoru.")
        urllib.request.urlopen(GATE_CLOSE_URL)
        print("Čekám na úplné zavření závory a uklidnění obrazu...")
        time.sleep(2)
    except Exception as e:
        print(f"Chyba při ovládání závory: {e}")

    payload = json.dumps({"spz": spz_val, "cas": cas_val, "typ_mista": typ_mista_val})
    gate_server.publish(payload)

    print("Data uložena/odeslána do gate_server. Závora zavřena.")
    print("\n--- Připraveno pro další auto (FÁZE 1) ---")

    # Give a 3-second lockout before accepting new Phase 1 inputs to ignore departing car taillights
    set_phase(1, 3.0)


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

    # If the phase transition lock is active, inputs are ignored (but physical state is still tracked)
    inputs_locked = now < phase_transition_lock

    # Blue LED edge detection with debouncing & lockout check
    modra_stisknuta = False
    if blue_pixels > MIN_PIXELS:
        if not modra_sviti_predtem and (now - last_blue_press > DEBOUNCE_COOLDOWN):
            if not inputs_locked:
                modra_stisknuta = True
                last_blue_press = now
        modra_sviti_predtem = True
    else:
        modra_sviti_predtem = False

    # Green LED edge detection with debouncing & lockout check
    zelena_stisknuta = False
    if green_pixels > MIN_PIXELS:
        if not zelena_sviti_predtem and (now - last_green_press > DEBOUNCE_COOLDOWN):
            if not inputs_locked:
                zelena_stisknuta = True
                last_green_press = now
        zelena_sviti_predtem = True
    else:
        zelena_sviti_predtem = False


    # --- STATE MACHINE ---

    # FÁZE 1: Čekání na příjezd auta (Modrá LED)
    if faze == 1:
        if modra_stisknuta and not ocr_in_progress:
            print("\n--- FÁZE 1: Auto detekováno ---")
            ocr_in_progress = True
            cas = 0
            pocet_zelenych_misto = 0

            # Start heavy download and OCR thread in background
            threading.Thread(target=process_ocr_async, daemon=True).start()

            # Transition to FÁZE 2 with a 2.0 second lockout to ignore robot sequence noise
            set_phase(2, 6.0)
            print("\n--- FÁZE 2: Zadávání času ---")
            print("Zadávejte čas pomocí zelené LED (1x = +15s), potvrďte modrou LED.")

    # FÁZE 2: Zadávání času (Zelená = +15s, Modrá = Potvrdit)
    elif faze == 2:
        if zelena_stisknuta:
            cas += 15
            print(f"Přidáno +15s! Celkový čas: {cas} s")

        elif modra_stisknuta:
            print(f"2. Modrá LED stisknuta! Čas potvrzen na {cas} s.")
            print("\n--- FÁZE 4: Výběr typu místa ---")
            print("Vyberte typ místa zelenou LED (1x=elektro, 2x=invalida, jiné=normální), potvrďte modrou.")
            pocet_zelenych_misto = 0

            # Transition to FÁZE 4 with a 1.5 second lockout
            set_phase(4, 1.5)

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

            print(f"3. Modrá LED stisknuta! Místo potvrzeno: {typ_mista}")

            # Transition to FÁZE 5 to lock inputs while gate is operating
            set_phase(5, 0.0)

            threading.Thread(
                target=process_gate_and_commit,
                args=(spz, cas, typ_mista),
                daemon=True,
            ).start()

    # FÁZE 5: Závora v provozu (Bypassuje detekci LED dokud auto neodjede)
    elif faze == 5:
        pass


    # Live debug overlay directly on display
    debug_frame = image.copy()

    # Display lockout status in top right corner if active
    if inputs_locked:
        cv2.putText(debug_frame, "VSTUPY ZAMCENY", (debug_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(debug_frame, f"F: {faze}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(debug_frame, f"B: {blue_pixels} / {MIN_PIXELS}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(debug_frame, f"G: {green_pixels} / {MIN_PIXELS}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Gate Camera Feed", debug_frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
