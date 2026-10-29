import sys
import types
from pathlib import Path

import pytest

import export


def test_engine_path_naming(tmp_path):
    assert export.engine_path("yolo11n", "fp16", 640, tmp_path) == tmp_path / "yolo11n_fp16_640.engine"


def test_engine_path_defaults_to_models_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(export, "MODELS_DIR", tmp_path)
    assert export.engine_path("yolo11s", "int8", 320) == tmp_path / "yolo11s_int8_320.engine"


def test_parse_engine_name_round_trip():
    path = export.engine_path("yolo11n", "int8", 640, Path("models"))
    assert export.parse_engine_name(path) == ("yolo11n", "int8", 640)


def test_parse_engine_name_with_underscore_in_model():
    assert export.parse_engine_name("models/my_model_fp16_416.engine") == ("my_model", "fp16", 416)


@pytest.mark.parametrize("name", ["custom.engine", "yolo11n_fp32_640.engine", "yolo11n_fp16_640.onnx"])
def test_parse_engine_name_rejects_other_names(name):
    assert export.parse_engine_name(name) is None


def test_export_command():
    assert export.export_command("yolo11n", "fp16", 640) == (
        "python3 src/export.py --model yolo11n --precision fp16 --imgsz 640"
    )


def test_export_kwargs_fp16():
    assert export.export_kwargs("fp16", 640, None, quantize_api=True) == {"format": "engine", "imgsz": 640, "half": True}


def test_export_kwargs_int8_old_api():
    assert export.export_kwargs("int8", 640, "coco8.yaml", quantize_api=False) == {
        "format": "engine", "imgsz": 640, "data": "coco8.yaml", "int8": True,
    }


def test_export_kwargs_int8_new_api():
    assert export.export_kwargs("int8", 640, "coco8.yaml", quantize_api=True) == {
        "format": "engine", "imgsz": 640, "data": "coco8.yaml", "quantize": 8,
    }


def test_parse_args_defaults():
    args = export.parse_args([])
    assert (args.model, args.precision, args.imgsz, args.calib_data, args.force) == ("yolo11n", "fp16", 640, None, False)


@pytest.mark.parametrize("argv", [
    ["--imgsz", "630"],             # not a multiple of 32
    ["--imgsz", "0"],
    ["--precision", "fp32"],
    ["--model", "yolo11n.pt"],      # Review Focus 5: a file, not a model name
    ["--model", "weights/best"],
    ["--model", "yolo11n.PT"],      # any file extension, any case
    ["--model", "best.onnx"],
    ["--model", "yolo11n.engine"],
    ["--model", "weights\\best"],
])
def test_parse_args_rejects(argv):
    with pytest.raises(SystemExit):
        export.parse_args(argv)


def test_main_skips_existing_engine(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(export, "MODELS_DIR", tmp_path)
    (tmp_path / "yolo11n_fp16_640.engine").write_bytes(b"x")
    assert export.main(["--model", "yolo11n"]) == 0  # would fail importing ultralytics if it tried to build
    assert "already exists" in capsys.readouterr().out


def test_main_unknown_model_exits_1(monkeypatch, tmp_path, capsys):
    # e.g. the common typo "yolov11n": attempt_download_asset finds nothing and returns a path that doesn't exist.
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = lambda weights: pytest.fail("YOLO() should not be called without weights")
    downloads = types.ModuleType("ultralytics.utils.downloads")
    downloads.attempt_download_asset = lambda path: path
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", types.ModuleType("ultralytics.utils"))
    monkeypatch.setitem(sys.modules, "ultralytics.utils.downloads", downloads)
    monkeypatch.setattr(export, "MODELS_DIR", tmp_path)

    assert export.main(["--model", "yolov11n"]) == 1
    out = capsys.readouterr().out
    assert "Could not find or download 'yolov11n.pt'" in out


@pytest.mark.parametrize("model, data", [
    ("yolo11n", "coco8.yaml"),
    ("yolo11s-seg", "coco8-seg.yaml"),
    ("yolo11n-pose", "coco8-pose.yaml"),
])
def test_default_calib_data_matches_task(model, data):
    assert export.default_calib_data(model) == data


def test_seg_engine_name_round_trips():
    path = export.engine_path("yolo11n-seg", "int8", 640, models_dir="m")
    assert path.name == "yolo11n-seg_int8_640.engine"
    assert export.parse_engine_name(path) == ("yolo11n-seg", "int8", 640)
