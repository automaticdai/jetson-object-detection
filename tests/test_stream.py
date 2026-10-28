import socket
import threading
import urllib.request

import numpy as np
import pytest

import detect
import stream


@pytest.fixture
def server():
    s = stream.MjpegServer(0, host="127.0.0.1").start()
    yield s
    s.stop()


def test_page_embeds_stream(server):
    html = urllib.request.urlopen(f"http://127.0.0.1:{server.port}/", timeout=5).read()
    assert b'<img src="/stream.mjpg"' in html


def test_unknown_path_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"http://127.0.0.1:{server.port}/nope", timeout=5)
    assert e.value.code == 404


def test_stream_sends_published_frames_as_jpeg(server):
    server.publish(np.zeros((48, 64, 3), np.uint8))
    resp = urllib.request.urlopen(f"http://127.0.0.1:{server.port}/stream.mjpg", timeout=5)
    assert resp.headers["Content-Type"] == "multipart/x-mixed-replace; boundary=frame"
    head = resp.read(200)
    assert head.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    assert b"\r\n\r\n\xff\xd8" in head  # JPEG start-of-image marker right after the part headers
    resp.close()


def test_frame_buffer_waits_for_a_new_frame():
    buf = stream.FrameBuffer()
    assert buf.wait_next(0, timeout=0.01) == (0, None)  # nothing published yet
    threading.Timer(0.02, buf.publish, args=(b"jpeg",)).start()
    assert buf.wait_next(0, timeout=2) == (1, b"jpeg")


def test_port_in_use_raises_oserror(server):
    with pytest.raises(OSError):
        stream.MjpegServer(server.port, host="127.0.0.1")


def test_stream_args():
    args = detect.parse_args(["--engine", "m.engine", "--stream", "8080"])
    assert (args.stream, args.stream_host) == (8080, "0.0.0.0")
    assert detect.parse_args(["--engine", "m.engine"]).stream is None


def test_stream_url_uses_hostname_for_all_interfaces():
    assert detect.stream_url("0.0.0.0", 8080) == f"http://{socket.gethostname()}:8080/"
    assert detect.stream_url("127.0.0.1", 9000) == "http://127.0.0.1:9000/"
