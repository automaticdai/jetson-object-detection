"""Build a TensorRT engine for a YOLO model with Ultralytics.

Engines only run on the device and TensorRT version they were built with, so run this on the Jetson itself.

Examples:
    python3 src/export.py                                   # models/yolo11n_fp16_640.engine
    python3 src/export.py --precision int8                  # INT8, calibrated on coco8 (small, quick)
    python3 src/export.py --model yolo11s --precision int8 --calib-data coco128.yaml
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_CALIB_DATA = "coco8.yaml"
_ENGINE_NAME = re.compile(r"^(?P<model>.+)_(?P<precision>fp16|int8)_(?P<imgsz>\d+)\.engine$")


def engine_path(model, precision, imgsz, models_dir=None):
    return Path(models_dir or MODELS_DIR) / f"{model}_{precision}_{imgsz}.engine"


def parse_engine_name(path):
    match = _ENGINE_NAME.match(Path(path).name)
    if match is None:
        return None
    return match["model"], match["precision"], int(match["imgsz"])


def export_command(model, precision, imgsz):
    return f"python3 src/export.py --model {model} --precision {precision} --imgsz {imgsz}"


def export_kwargs(precision, imgsz, calib_data, quantize_api):
    kwargs = {"format": "engine", "imgsz": imgsz}
    if precision == "fp16":
        kwargs["half"] = True
    else:
        kwargs["data"] = calib_data
        # Newer Ultralytics versions replaced int8=True with quantize=8.
        if quantize_api:
            kwargs["quantize"] = 8
        else:
            kwargs["int8"] = True
    return kwargs


def supports_quantize_arg():
    from ultralytics.utils import DEFAULT_CFG_DICT

    return "quantize" in DEFAULT_CFG_DICT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo11n", help="Ultralytics model name, e.g. yolo11n, yolo11s (default: yolo11n)")
    parser.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--imgsz", type=int, default=640, help="input size, a multiple of 32 (default: 640)")
    parser.add_argument("--calib-data", default=None, help=f"INT8 calibration dataset yaml (default: {DEFAULT_CALIB_DATA})")
    parser.add_argument("--force", action="store_true", help="rebuild even if the engine already exists")
    args = parser.parse_args(argv)
    if "/" in args.model or args.model.endswith(".pt"):
        parser.error(f"--model takes a model name such as yolo11n, not a file path (got '{args.model}')")
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error(f"--imgsz must be a positive multiple of 32 (got {args.imgsz})")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = engine_path(args.model, args.precision, args.imgsz)
    if out.exists() and not args.force:
        print(f"{out} already exists; use --force to rebuild it.")
        return 0

    calib_data = args.calib_data
    if args.precision == "int8" and calib_data is None:
        calib_data = DEFAULT_CALIB_DATA
        print(f"Warning: no --calib-data given; calibrating on {calib_data}, which is tiny and may reduce INT8 accuracy.")

    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    out.parent.mkdir(parents=True, exist_ok=True)
    weights = attempt_download_asset(str(out.parent / f"{args.model}.pt"))  # no-op if already downloaded
    print(f"Building {out.name}. This can take several minutes...")
    kwargs = export_kwargs(args.precision, args.imgsz, calib_data, supports_quantize_arg())
    built = YOLO(weights).export(**kwargs)
    shutil.move(built, out)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
