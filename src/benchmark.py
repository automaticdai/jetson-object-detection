"""Benchmark TensorRT engines on the same frames and print a comparison table.

Frames are loaded into memory first, so camera and video decoding don't affect the numbers;
each engine gets the same warm-up and the same frames.

Examples:
    python3 src/benchmark.py                                   # every engine in models/, on Ultralytics' bus.jpg
    python3 src/benchmark.py --engines models/yolo11n_fp16_640.engine models/yolo11n_int8_640.engine
    python3 src/benchmark.py --source video.mp4 --frames 500 --csv results/orin.csv
    python3 src/benchmark.py --backend tensorrt --source bus.jpg     # Jetson Nano (no Ultralytics)
"""
import argparse
import csv
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2 as cv

from detect import load_model, resolve_backend
from export import MODELS_DIR

STAGES = ("preprocess", "inference", "postprocess")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", type=Path, nargs="+", default=None, help="engines to compare (default: models/*.engine)")
    parser.add_argument("--source", default=None, help="image or video to run on (default: Ultralytics' bus.jpg)")
    parser.add_argument("--frames", type=int, default=300, help="timed frames per engine (default: 300)")
    parser.add_argument("--warmup", type=int, default=30, help="untimed frames per engine first (default: 30)")
    parser.add_argument("--backend", choices=["auto", "ultralytics", "tensorrt"], default="auto",
                        help="as in detect.py (default: auto)")
    parser.add_argument("--csv", type=Path, default=None, help="also write the results to this CSV file")
    args = parser.parse_args(argv)
    if args.frames <= 0 or args.warmup < 0:
        parser.error("--frames must be positive and --warmup must not be negative")
    return args


def default_engines(models_dir=None):
    return sorted(Path(models_dir or MODELS_DIR).glob("*.engine"))


def default_source():
    from ultralytics.utils import ASSETS

    return str(ASSETS / "bus.jpg")


def load_frames(source, count):
    """Read `count` frames from an image or video, looping the video if it is shorter."""
    cap = cv.VideoCapture(source)
    frames = []
    while len(frames) < count:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Cannot read any frames from '{source}'.")
    return [frames[i % len(frames)] for i in range(count)]


def percentile(values, p):
    """Nearest-rank percentile, p in 0-100."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(p / 100 * len(ordered)) - 1))
    return ordered[index]


def summarize(name, latencies_ms, stage_ms):
    total_s = sum(latencies_ms) / 1e3
    row = {"engine": name, "fps": len(latencies_ms) / total_s if total_s > 0 else 0.0}
    row.update({stage: sum(v) / len(v) for stage, v in stage_ms.items()})
    row["p50_ms"] = percentile(latencies_ms, 50)
    row["p95_ms"] = percentile(latencies_ms, 95)
    return row


def run_engine(engine, frames, warmup, backend="ultralytics"):
    model = load_model(engine, backend)
    for frame in frames[:warmup]:
        model.predict(frame, verbose=False)
    latencies, stage_ms = [], {stage: [] for stage in STAGES}
    for frame in frames:
        start = time.perf_counter()
        result = model.predict(frame, verbose=False)[0]
        latencies.append((time.perf_counter() - start) * 1e3)
        for stage in STAGES:
            stage_ms[stage].append(result.speed[stage])
    return summarize(Path(engine).name, latencies, stage_ms)


def device_info():
    """e.g. 'NVIDIA Jetson AGX Orin Developer Kit, L4T R36.4.0, power mode MAXN'."""
    parts = []
    try:
        parts.append(Path("/proc/device-tree/model").read_text().rstrip("\x00\n"))
    except OSError:
        parts.append(f"{platform.node()} ({platform.machine()})")
    try:
        first = Path("/etc/nv_tegra_release").read_text().splitlines()[0]  # "# R36 (release), REVISION: 4.0, ..."
        release, revision = first.split(",")[:2]
        parts.append(f"L4T {release.split()[1]}.{revision.split(':')[1].strip()}")
    except (OSError, IndexError, ValueError):
        pass
    try:
        out = subprocess.run(  # no capture_output/text: Python 3.6 on the Jetson Nano
            ["nvpmodel", "-q"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=5
        ).stdout
        mode = next(line.split(":", 1)[1].strip() for line in out.splitlines() if "Power Mode" in line)
        parts.append(f"power mode {mode}")
    except (OSError, subprocess.SubprocessError, StopIteration):
        pass
    return ", ".join(parts)


def format_table(rows):
    header = ["Engine", "FPS", "Pre (ms)", "Infer (ms)", "Post (ms)", "p50 (ms)", "p95 (ms)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        values = [r["engine"], f"{r['fps']:.1f}"] + [f"{r[k]:.1f}" for k in (*STAGES, "p50_ms", "p95_ms")]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_csv(path, rows, device):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["device", *rows[0].keys()])
        writer.writeheader()
        for row in rows:
            writer.writerow({"device": device, **{k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}})


def main(argv=None):
    args = parse_args(argv)
    engines = args.engines or default_engines()
    if not engines:
        print("No engines found in models/. Build one first, e.g.:\n    python3 src/export.py --model yolo11n --precision fp16")
        return 1
    missing = [str(e) for e in engines if not Path(e).exists()]
    if missing:
        print(f"Engine(s) not found: {', '.join(missing)}")
        return 1

    backend = resolve_backend(args.backend)
    if args.source is None and backend == "tensorrt":
        print("The tensorrt backend has no bundled test image; pass --source IMAGE_OR_VIDEO.")
        return 1
    source = args.source or default_source()
    try:
        frames = load_frames(source, args.frames)
    except ValueError as e:
        print(e)
        return 1

    device = device_info()
    print(f"Device: {device}")
    print(f"Source: {source}, {args.frames} frames after {args.warmup} warm-up frames per engine\n", flush=True)
    rows = []
    for engine in engines:
        print(f"Running {Path(engine).name}...", flush=True)
        try:
            rows.append(run_engine(engine, frames, args.warmup, backend))
        except RuntimeError as e:  # e.g. built on another device
            print(e)
            return 1

    print()
    print(format_table(rows))
    if args.csv:
        write_csv(args.csv, rows, device)
        print(f"\nSaved {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
