import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- SHARED STATE & THREAD LOCK ---
_data_lock = threading.Lock()
_current_data = {
    "spz": "",
    "cas": 0,
    "typ_mista": "",
    "timestamp": None,
    "status": "idle",
}


# --- HTTP REQUEST HANDLER ---
class GateServerRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Serves current gate state as JSON."""
        if self.path in ("/get_last_spz", "/data", "/status"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with _data_lock:
                response_payload = json.dumps(_current_data, ensure_ascii=False, indent=2).encode("utf-8")

            self.wfile.write(response_payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppresses default HTTP request logging to keep console clean."""
        return


# --- MODULE API ---
def start(host="0.0.0.0", port=8080):
    """Starts the HTTP server on a background daemon thread."""
    server = HTTPServer((host, port), GateServerRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[gate_server] Listening for requests on http://{host}:{port}/data")


def publish(data):
    """Updates stored state.

    Accepts:
    - Dict: `{"spz": "1AB2345", "cas": 30, "typ_mista": "elektro"}`
    - JSON string representation of a dict
    - Raw SPZ string (initial detection step)
    """
    global _current_data
    now_iso = datetime.now().isoformat()

    with _data_lock:
        if isinstance(data, dict):
            _current_data.update(data)
            _current_data["timestamp"] = now_iso
            _current_data["status"] = "confirmed"

        elif isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    _current_data.update(parsed)
                    _current_data["timestamp"] = now_iso
                    _current_data["status"] = "confirmed"
                else:
                    _current_data["spz"] = str(parsed)
                    _current_data["timestamp"] = now_iso
                    _current_data["status"] = "detected"
            except json.JSONDecodeError:
                _current_data["spz"] = data
                _current_data["timestamp"] = now_iso
                _current_data["status"] = "detected"


def get_data():
    """Returns a thread-safe copy of the current state dict."""
    with _data_lock:
        return _current_data.copy()
