import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

import camera


def test_gstreamer_pipeline_uses_settings():
    p = camera.gstreamer_pipeline(sensor_id=1, width=640, height=480, fps=15, flip=2)
    assert p.startswith("nvarguscamerasrc sensor-id=1 ! ")
    assert "width=640,height=480,framerate=15/1" in p
    assert "flip-method=2" in p
    assert p.endswith("appsink drop=1")


def test_add_camera_args_defaults():
    parser = argparse.ArgumentParser()
    camera.add_camera_args(parser)
    args = parser.parse_args([])
    assert (args.source, args.sensor_id, args.device) == ("csi", 0, 0)
    assert (args.width, args.height, args.fps, args.flip) == (1280, 720, 30, 0)


def test_csi_without_gstreamer_raises(monkeypatch):
    monkeypatch.setattr(camera, "has_gstreamer", lambda: False)
    with pytest.raises(camera.CameraError, match="GStreamer"):
        camera.open_capture("csi")


def test_missing_file_raises(tmp_path):
    with pytest.raises(camera.CameraError, match="nope.mp4"):
        camera.open_capture(str(tmp_path / "nope.mp4"))


def test_opens_video_file(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 40, np.uint8))
    writer.release()

    cap = camera.open_capture(str(path))
    try:
        ok, frame = cap.read()
        assert ok and frame.shape == (48, 64, 3)
    finally:
        cap.release()


def test_open_capture_from_args_passes_source(monkeypatch):
    seen = {}
    monkeypatch.setattr(camera, "open_capture", lambda source, **kw: seen.update(source=source, **kw) or "cap")
    args = argparse.Namespace(source="usb", sensor_id=0, device=2, width=640, height=480, fps=60, flip=0)
    assert camera.open_capture_from_args(args) == "cap"
    assert seen == dict(source="usb", sensor_id=0, device=2, width=640, height=480, fps=60, flip=0)


def test_has_gui_false_on_headless_opencv():
    # requirements-dev.txt installs opencv-python-headless, like the Ultralytics Jetson image does.
    assert camera.has_gui() is False


def test_window_error_none_when_headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    assert camera.window_error(headless=True) is None


def test_window_error_no_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    assert camera.window_error(headless=False) == camera.NO_DISPLAY_MSG


def test_window_error_no_gui(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    assert camera.window_error(headless=False) == camera.NO_GUI_MSG


def test_camera_test_without_gui_exits_cleanly(tmp_path):
    script = Path(__file__).resolve().parent.parent / "src" / "camera_test.py"
    env = {**os.environ, "DISPLAY": ":0"}
    proc = subprocess.run(
        [sys.executable, str(script), "--source", str(tmp_path / "x.mp4")], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 1
    assert "no GUI support" in proc.stdout
    assert "Traceback" not in proc.stderr


def test_usb_requests_mjpg_before_resolution(monkeypatch):
    calls = []

    class RecordingCapture:
        def __init__(self, *args):
            pass

        def set(self, prop, value):
            calls.append((prop, value))
            return True

        def isOpened(self):
            return True

    monkeypatch.setattr(camera.cv, "VideoCapture", RecordingCapture)
    camera.open_capture("usb", width=1280, height=720)
    props = [prop for prop, _ in calls]
    assert calls[0] == (cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    assert props.index(cv.CAP_PROP_FOURCC) < props.index(cv.CAP_PROP_FRAME_WIDTH)
