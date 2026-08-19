import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_lock = threading.Lock()
_spz = None
_server = None
_thread = None


class _GridHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        match self.path:
            case "/get_last_spz":
                with _lock:
                    body = json.dumps({"spz": _spz}).encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                _ = self.wfile.write(body)
            case _:
                self.send_error(404)

    #def log_message(self, *args):
    #    pass  # remove console logs


def start(host="0.0.0.0", port=8000):
    global _server, _thread
    _server = ThreadingHTTPServer((host, port), _GridHandler)
    _thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _thread.start()


def publish(spz):
    global _spz
    with _lock:
        _spz = spz


def stop():
    if _server:
        _server.shutdown()
