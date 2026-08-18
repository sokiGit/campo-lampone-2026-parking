import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from urllib.parse import unquote

_lock = threading.Lock()
_latest_cells = []
_latest_car_positions = []
_car_tracking = {}
_server = None
_thread = None

def filter_dict_keys(data, allowed_keys: set):
    if isinstance(data, list):
        return [filter_dict_keys(item, allowed_keys) for item in data]

    if isinstance(data, dict):
        if any(isinstance(v, dict) for v in data.values()):
            return {
                outer_k: {k: inner_v[k] for k in allowed_keys if k in inner_v}
                if isinstance(inner_v, dict) else inner_v
                for outer_k, inner_v in data.items()
            }
        return {k: data[k] for k in allowed_keys if k in data}

    return data

class _GridHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        match self.path:
            case "/debug":
                with _lock:
                    body = json.dumps({"cells": _latest_cells, "your_position": _latest_car_positions}).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                _ = self.wfile.write(body)
            case _:
                if self.path.startswith("/get_my_data?spz="):
                    spz = unquote(self.path[len("/get_my_data?spz="):])
                    ct_data = None

                    for _, data in _car_tracking.items():
                        if data.get("spz") == spz:
                            ct_data = data
                            break

                    ct_data_filtered = filter_dict_keys(ct_data, {"pos_px", "pos_grid", "angle", "spz"})
                    body = json.dumps({
                        "car_tracking_data": ct_data_filtered,
                        "cells": _latest_cells,
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    _ = self.wfile.write(body)
                else:
                    self.send_error(404)

    #def log_message(self, *args):
    #    pass  # remove console logs


def start(host="0.0.0.0", port=8000):
    global _server, _thread
    _server = ThreadingHTTPServer((host, port), _GridHandler)
    _thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _thread.start()


def publish(cells, car_positions, car_tracking):
    global _latest_cells, _latest_car_positions, _car_tracking
    with _lock:
        _latest_cells = [dict(c) for c in cells]
        _latest_car_positions = [dict(c) for c in car_positions]
        _car_tracking = dict(car_tracking)


def stop():
    if _server:
        _server.shutdown()
