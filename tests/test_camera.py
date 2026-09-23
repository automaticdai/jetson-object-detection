import argparse

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
