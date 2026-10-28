"""Live YOLO object detection on Jetson with a TensorRT engine.

Build an engine on this device first:
    python3 src/export.py --model yolo11n --precision fp16

Examples:
    python3 src/detect.py --engine models/yolo11n_fp16_640.engine
    python3 src/detect.py --engine models/yolo11n_fp16_640.engine --source usb --classes person,car --conf 0.5
    python3 src/detect.py --engine models/yolo11n_int8_640.engine --source video.mp4 --save out.mp4 --headless
"""
import argparse
import math
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from camera import CameraError, add_camera_args, open_capture_from_args, window_error
from export import export_command, parse_engine_name
from timing import StageTimer

DEFAULT_FPS = 30.0


def conf_value(text):
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, e.g. 0.5 for 50% (got {text})")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", type=Path, required=True, help="TensorRT engine built by src/export.py")
    add_camera_args(parser)
    parser.add_argument("--conf", type=conf_value, default=0.25, help="confidence threshold, 0-1 (default: 0.25)")
    parser.add_argument("--classes", default=None, help="comma-separated class names, e.g. person,car (default: all)")
    parser.add_argument("--save", type=Path, default=None, help="save the annotated video to this .mp4 file")
    parser.add_argument("--headless", action="store_true", help="do not open a window; print FPS and timings instead")
    return parser.parse_args(argv)


def resolve_classes(spec, model_names):
    """Turn 'person, Car' into class ids using the model's {id: name} map. None means all classes."""
    if spec is None:
        return None
    lookup = {name.lower(): idx for idx, name in model_names.items()}
    ids, unknown = [], []
    for name in (part.strip().lower() for part in spec.split(",")):
        if not name:
            continue
        if name in lookup:
            ids.append(lookup[name])
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"Unknown class name(s): {', '.join(unknown)}.\nValid names: {', '.join(model_names.values())}"
        )
    return ids or None


def source_fps(reported):
    # Cameras often report 0, NaN or nonsense (e.g. 90000) for CAP_PROP_FPS.
    return reported if math.isfinite(reported) and 0 < reported <= 240 else DEFAULT_FPS


def open_writer(path, frame_size, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise OSError(f"Cannot write video to {path}")
    return writer


def overlay_lines(timer):
    lines = [f"{timer.fps():.1f} FPS"]
    lines += [f"{stage}: {ms:.1f} ms" for stage, ms in timer.averages().items()]
    return lines


def draw_overlay(frame, lines):
    for i, line in enumerate(lines):
        cv.putText(frame, line, (10, 30 + i * 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def missing_engine_message(engine):
    parsed = parse_engine_name(engine)
    command = export_command(*parsed) if parsed else "python3 src/export.py --model yolo11n --precision fp16"
    return f"Engine not found: {engine}\nBuild it on this device first:\n    {command}"


def load_model(engine):
    from ultralytics import YOLO

    model = YOLO(str(engine), task="detect")
    try:
        # The engine is loaded lazily on the first predict; do it now so the first real frame's timing is honest.
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {engine}: {e}\n"
            "Engines only run on the device and TensorRT version they were built with. "
            "Re-export it on this device with src/export.py --force."
        ) from e
    return model


def main(argv=None):
    args = parse_args(argv)

    error = window_error(args.headless)
    if error:
        print(error)
        return 1
    if not args.engine.exists():
        print(missing_engine_message(args.engine))
        return 1

    try:
        model = load_model(args.engine)
        classes = resolve_classes(args.classes, model.names)
        cap = open_capture_from_args(args)
    except (RuntimeError, ValueError) as e:  # CameraError is a RuntimeError
        print(e)
        return 1

    timer = StageTimer()
    writer = None
    fps_out = source_fps(cap.get(cv.CAP_PROP_FPS))
    last_print = time.perf_counter()
    try:
        while True:
            with timer.measure("capture"):
                ret, frame = cap.read()
            if not ret:
                print("End of stream or camera disconnected.")
                break

            result = model.predict(frame, conf=args.conf, classes=classes, verbose=False)[0]
            for stage, ms in result.speed.items():  # preprocess, inference, postprocess
                timer.add(stage, ms)

            with timer.measure("draw"):
                annotated = result.plot()
                draw_overlay(annotated, overlay_lines(timer))
            timer.tick()

            if args.save:
                if writer is None:
                    writer = open_writer(args.save, (annotated.shape[1], annotated.shape[0]), fps_out)
                writer.write(annotated)

            if args.headless:
                now = time.perf_counter()
                if now - last_print >= 1.0:
                    print(" | ".join(overlay_lines(timer)), flush=True)
                    last_print = now
            else:
                cv.imshow("Detection", annotated)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print(e)
        return 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
