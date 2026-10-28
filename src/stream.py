"""Serve frames as an MJPEG stream over HTTP, so any browser can watch a headless Jetson."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2 as cv

BOUNDARY = "frame"
PAGE = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>jetson-object-detection</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100vh}</style></head>
<body><img src="/stream.mjpg" alt="live detection"></body></html>
"""


class FrameBuffer:
    """Latest JPEG plus a sequence number, so each client waits for a frame it has not sent yet."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._seq = 0
        self.closed = False

    def publish(self, jpeg):
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def close(self):
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    def wait_next(self, seq, timeout=1.0):
        """Return (seq, jpeg) of the next frame after `seq`, or the same seq on timeout."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != seq or self.closed, timeout=timeout)
            return self._seq, self._jpeg


def _make_handler(buffer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seq = 0
                try:
                    while not buffer.closed:
                        new_seq, jpeg = buffer.wait_next(seq)
                        if new_seq == seq or jpeg is None:
                            continue
                        seq = new_seq
                        self.wfile.write(
                            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the browser tab was closed
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass  # keep the console for FPS output

    return Handler


class MjpegServer:
    def __init__(self, port, host="0.0.0.0", quality=80):
        self.quality = quality
        self.buffer = FrameBuffer()
        self._server = ThreadingHTTPServer((host, port), _make_handler(self.buffer))
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self):
        return self._server.server_address[1]

    def start(self):
        self._thread.start()
        return self

    def publish(self, frame):
        ok, jpeg = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            self.buffer.publish(jpeg.tobytes())

    def stop(self):
        self.buffer.close()
        self._server.shutdown()
        self._server.server_close()
