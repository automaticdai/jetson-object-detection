import math

import cv2 as cv
import numpy as np
import pytest

import detect
from timing import StageTimer

NAMES = {0: "person", 1: "bicycle", 2: "car"}


def test_resolve_classes_none_means_all():
    assert detect.resolve_classes(None, NAMES) is None
    assert detect.resolve_classes("  ", NAMES) is None
    assert detect.resolve_classes(",", NAMES) is None


def test_resolve_classes_by_name():
    assert detect.resolve_classes("person,car", NAMES) == [0, 2]


def test_resolve_classes_ignores_spaces_and_case():  # Review Focus 1
    assert detect.resolve_classes(" Person , CAR ", NAMES) == [0, 2]


def test_resolve_classes_unknown_lists_valid_names():
    with pytest.raises(ValueError) as e:
        detect.resolve_classes("person,dog", NAMES)
    assert "dog" in str(e.value)
    assert "person, bicycle, car" in str(e.value)


@pytest.mark.parametrize("reported, expected", [(25.0, 25.0), (0.0, 30.0), (math.nan, 30.0), (90000.0, 30.0)])
def test_source_fps_fallback(reported, expected):  # Review Focus 3
    assert detect.source_fps(reported) == expected


def test_open_writer_creates_folder(tmp_path):  # Review Focus 2
    path = tmp_path / "results" / "run1" / "out.mp4"
    writer = detect.open_writer(path, (64, 48), 30.0)
    writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    assert path.exists() and path.stat().st_size > 0


def test_overlay_lines():
    t = StageTimer()
    t.add("capture", 1.0)
    t.add("inference", 7.25)
    t.tick(now=0.0)
    t.tick(now=0.05)
    assert detect.overlay_lines(t) == ["20.0 FPS", "capture: 1.0 ms", "inference: 7.2 ms"]


def test_missing_engine_message_gives_exact_command(tmp_path):
    msg = detect.missing_engine_message(tmp_path / "yolo11s_int8_320.engine")
    assert "python3 src/export.py --model yolo11s --precision int8 --imgsz 320" in msg


def test_missing_engine_message_for_custom_name(tmp_path):
    msg = detect.missing_engine_message(tmp_path / "custom.engine")
    assert "python3 src/export.py --model yolo11n --precision fp16" in msg


@pytest.mark.parametrize("value", ["50", "0", "-0.1", "1.5"])
def test_conf_out_of_range_rejected(value):  # Review Focus 4
    with pytest.raises(SystemExit):
        detect.parse_args(["--engine", "x.engine", "--conf", value])


def test_parse_args_defaults():
    args = detect.parse_args(["--engine", "m.engine"])
    assert (args.conf, args.classes, args.save, args.headless, args.source) == (0.25, None, None, False, "csi")


def test_main_without_gui_exits_1_before_loading(tmp_path, monkeypatch, capsys):
    # The Ultralytics Jetson image ships headless OpenCV; opening a window would crash on the first frame.
    monkeypatch.setenv("DISPLAY", ":0")
    engine = tmp_path / "yolo11n_fp16_640.engine"
    engine.write_bytes(b"x")
    monkeypatch.setattr(detect, "load_model", lambda e: pytest.fail("should exit before loading the engine"))
    assert detect.main(["--engine", str(engine)]) == 1
    assert "no GUI support" in capsys.readouterr().out


def test_main_missing_engine_exits_1(tmp_path, capsys):
    rc = detect.main(["--engine", str(tmp_path / "yolo11n_fp16_640.engine"), "--headless"])
    assert rc == 1
    assert "python3 src/export.py --model yolo11n --precision fp16 --imgsz 640" in capsys.readouterr().out


def test_track_flag():
    assert detect.parse_args(["--engine", "m.engine"]).track is None
    assert detect.parse_args(["--engine", "m.engine", "--track"]).track == "bytetrack"
    assert detect.parse_args(["--engine", "m.engine", "--track", "botsort"]).track == "botsort"
    with pytest.raises(SystemExit):
        detect.parse_args(["--engine", "m.engine", "--track", "sort"])


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(("predict", kwargs))
        return ["result"]

    def track(self, frame, **kwargs):
        self.calls.append(("track", kwargs))
        return ["result"]


def test_run_model_predicts_without_track():
    model = FakeModel()
    args = detect.parse_args(["--engine", "m.engine", "--conf", "0.4"])
    assert detect.run_model(model, None, args, [0]) == "result"
    assert model.calls == [("predict", {"conf": 0.4, "classes": [0], "verbose": False})]


def test_run_model_tracks_with_persistent_state():
    model = FakeModel()
    args = detect.parse_args(["--engine", "m.engine", "--track", "botsort"])
    detect.run_model(model, None, args, None)
    method, kwargs = model.calls[0]
    assert method == "track"
    assert kwargs["persist"] is True and kwargs["tracker"] == "botsort.yaml"
