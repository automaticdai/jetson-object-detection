# TensorRT Detection Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Live YOLO object detection on Jetson with Ultralytics TensorRT engines (FP16/INT8), with per-stage timing, class filtering and video saving, run from a Docker container.

**Architecture:** Small single-purpose modules under `src/`: `camera.py` (frame sources), `timing.py` (rolling stage timer), `export.py` (engine builder), and `detect.py` (the demo loop). Each is run as `python3 src/<name>.py`. Ultralytics is imported only inside functions, so all unit tests run on a PC without a GPU. `docker/run.sh` starts the Ultralytics Jetson image with the camera and display passed through.

**Tech Stack:** Python 3.10+, OpenCV 4.x, Ultralytics YOLO (TensorRT export), Docker (`ultralytics/ultralytics` Jetson images), pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-trt-detection-demo-design.md`

## Global Constraints

- Python 3.10+ (JetPack 6 system Python).
- **Never `pip install opencv-python*` on the Jetson host.** It shadows JetPack's GStreamer-enabled OpenCV. `requirements-dev.txt` is for PC/CI test runs only.
- `ultralytics` is imported **only inside functions**, never at module top level.
- Engine files are named `models/<model>_<precision>_<imgsz>.engine`, where `precision` is `fp16` or `int8`.
- Docker images: L4T R36 (JetPack 6, Orin) → `ultralytics/ultralytics:latest-jetson-jetpack6`; L4T R38 (JetPack 7.0, Thor) → `ultralytics/ultralytics:latest-nvidia-arm64`. Pinned to a version in Task 1.
- Every user-facing failure prints a plain message (no traceback) and exits with a non-zero code.
- Repo code is MIT. The README must say that Ultralytics is AGPL-3.0.
- Scripts are run from the repo root: `python3 src/<script>.py`. Modules import each other by bare name (`from camera import ...`).
- Tests run from the repo root with `python3 -m pytest tests -v`.

## Review Focus

1. `--classes " Person , CAR "` (spaces, different case) should select person and car, not fail as unknown. Test in Task 5.
2. `--save results/run1/out.mp4` where `results/run1/` doesn't exist should create the folder, not fail silently. Test in Task 5.
3. CSI/USB sources often report 0 or NaN FPS. The saved video should use 30 FPS, not produce a broken file. Test in Task 5.
4. `--conf 50` (meaning 50%) should be rejected with a clear error, not accepted as a threshold that filters out every detection. Test in Task 5.
5. `--model yolo11n.pt` or `--model weights/best.pt` should be rejected, telling the user to pass a model name. Otherwise it would produce engine names like `yolo11n.pt_fp16_640.engine`. Test in Task 4.

## File Map

| File | Responsibility |
|---|---|
| `docker/run.sh` (new) | Choose the image for this JetPack version; run a command in it with the GPU, cameras, Argus socket and X11 passed through |
| `.gitignore` (modify) | Ignore `models/`, weights, engines, videos |
| `src/camera.py` (new) | `CameraError`, `gstreamer_pipeline`, `has_gstreamer`, `open_capture`, `open_capture_from_args`, `add_camera_args`, `display_available`, `NO_DISPLAY_MSG` |
| `src/camera_test.py` (modify) | Camera preview script, now using `camera.py` |
| `src/timing.py` (new) | `StageTimer`: rolling per-stage ms and FPS |
| `src/export.py` (new) | Engine naming/parsing, export kwargs, CLI that builds engines |
| `src/detect.py` (new) | Detection demo CLI: model loading, class resolution, video writer, overlay, main loop |
| `tests/conftest.py` (new) | Puts `src/` on `sys.path` |
| `tests/test_camera.py`, `tests/test_timing.py`, `tests/test_export.py`, `tests/test_detect.py` (new) | Unit tests |
| `requirements-dev.txt` (new) | pytest + headless OpenCV, for PC/CI only |
| `README.md` (modify) | Detection section, Docker usage, AGPL note, FPS table |

Tasks 1 and 6 need **the Jetson (AGX Orin) and a person at it**. Tasks 2–5 run anywhere.

---

### Task 1: Docker runner + CSI-in-container risk check  **[needs Jetson]**

**Files:**
- Create: `docker/run.sh`
- Modify: `.gitignore` (append)

**Interfaces:**
- Consumes: the existing `src/camera_test.py` (already supports `--source csi|usb|<file>` and `--headless`).
- Produces: `docker/run.sh [--image IMAGE] [command...]`, used by the README and by Task 6.

- [ ] **Step 1: Write `docker/run.sh`**

```bash
#!/usr/bin/env bash
# Run a command inside the Ultralytics Jetson container with GPU, camera and display access.
#
# Usage: docker/run.sh [--image IMAGE] [command...]      (default command: bash)
# Example: docker/run.sh python3 src/detect.py --engine models/yolo11n_fp16_640.engine
set -euo pipefail
shopt -s nullglob

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE=""
if [[ "${1:-}" == "--image" ]]; then
    IMAGE="${2:?--image needs a value}"
    shift 2
fi

if [[ -z "$IMAGE" ]]; then
    # First line looks like: "# R36 (release), REVISION: 4.0, ..."
    release="$(head -n1 /etc/nv_tegra_release 2>/dev/null | sed -n 's/^# R\([0-9]*\).*/\1/p')"
    case "$release" in
        36) IMAGE="ultralytics/ultralytics:latest-jetson-jetpack6" ;;  # JetPack 6 (Orin)
        38) IMAGE="ultralytics/ultralytics:latest-nvidia-arm64" ;;     # JetPack 7.0 (Thor)
        *)
            echo "Unknown or unsupported Jetson Linux release 'R${release:-?}'. Pass --image IMAGE." >&2
            exit 1
            ;;
    esac
fi

args=(--rm -i --runtime nvidia --ipc=host -v "$REPO_DIR:/app" -w /app)
[[ -t 0 ]] && args+=(-t)

for dev in /dev/video*; do
    args+=(--device "$dev")
done

# CSI cameras are served by nvargus-daemon on the host through this socket.
if [[ -S /tmp/argus_socket ]]; then
    args+=(-v /tmp/argus_socket:/tmp/argus_socket)
fi

if [[ -n "${DISPLAY:-}" ]]; then
    args+=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix)
    echo "Display: $DISPLAY (if windows fail to open, run 'xhost +local:docker' on the host)" >&2
fi

if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker run "${args[@]}" "$IMAGE" "$@"
```

Then: `chmod +x docker/run.sh`

- [ ] **Step 2: Syntax check**

Run: `bash -n docker/run.sh && (command -v shellcheck >/dev/null && shellcheck docker/run.sh || echo "shellcheck not installed, skipped")`
Expected: no output from `bash -n`; shellcheck clean or skipped.

- [ ] **Step 3: Append to `.gitignore`**

```gitignore

# Models, engines and media (engines are device-specific; never commit them)
models/
*.pt
*.onnx
*.engine
*.mp4
*.avi
```

- [ ] **Step 4: [Jetson] Pull the image and record its Ultralytics version**

Run on the Orin:
```bash
docker/run.sh python3 -c "import ultralytics, cv2; print('ultralytics', ultralytics.__version__); print('opencv', cv2.__version__); print([l.strip() for l in cv2.getBuildInformation().splitlines() if 'GStreamer' in l])"
```
Expected: the versions are printed. Write down the Ultralytics version and whether GStreamer is `YES` or `NO`.

- [ ] **Step 5: [Jetson] Risk check — CSI camera inside the container**

Run (with a monitor or X forwarding, after `xhost +local:docker`):
```bash
docker/run.sh python3 src/camera_test.py --source csi
docker/run.sh python3 src/camera_test.py --source csi --headless   # Ctrl+C after a few FPS lines
docker/run.sh python3 src/camera_test.py --source usb --headless   # if a USB camera is attached
```
Expected: the CSI preview shows and prints ~30 FPS.
**Decision gate:** if the CSI camera fails with "no GStreamer support", **stop and report to the user**. Continue Tasks 2–5 (they don't depend on this), but Task 6 waits for the user to choose a fallback (spec, Risks §1).

- [ ] **Step 6: [Jetson] Pin the image**

Look for a versioned tag matching the version from Step 4 (e.g. `<version>-jetson-jetpack6`) on https://hub.docker.com/r/ultralytics/ultralytics/tags. If there is one, replace `latest-jetson-jetpack6` in `run.sh` with it. If not, pin by digest:
```bash
docker inspect --format '{{index .RepoDigests 0}}' ultralytics/ultralytics:latest-jetson-jetpack6
```
and use the printed `ultralytics/ultralytics@sha256:...` as the R36 image. Leave the R38 (Thor) line as `latest-nvidia-arm64`; it's untested.

- [ ] **Step 7: Commit**

```bash
git add docker/run.sh .gitignore
git commit -m "Add Docker runner for Ultralytics Jetson image; ignore models and media"
```

---

### Task 2: Shared camera module + test setup

**Files:**
- Create: `src/camera.py`, `tests/conftest.py`, `tests/test_camera.py`, `requirements-dev.txt`
- Modify: `src/camera_test.py` (whole file)

**Interfaces:**
- Produces (used by Tasks 5 and 6):
  - `class CameraError(RuntimeError)`
  - `NO_DISPLAY_MSG: str`
  - `gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30, flip=0) -> str`
  - `has_gstreamer() -> bool`
  - `open_capture(source: str, *, sensor_id=0, device=0, width=1280, height=720, fps=30, flip=0) -> cv.VideoCapture`, which raises `CameraError` if GStreamer is missing (CSI) or the source doesn't open
  - `open_capture_from_args(args: argparse.Namespace) -> cv.VideoCapture`
  - `add_camera_args(parser: argparse.ArgumentParser) -> None`, which adds `--source --sensor-id --device --width --height --fps --flip`
  - `display_available() -> bool`

- [ ] **Step 1: Test setup**

`requirements-dev.txt`:
```text
# For running unit tests on a PC or in CI only.
# Do NOT install these on the Jetson: pip OpenCV has no GStreamer and shadows JetPack's OpenCV.
pytest>=8
numpy
opencv-python-headless
```

`tests/conftest.py`:
```python
import sys
from pathlib import Path

# Scripts in src/ import each other by bare module name, as they do when run as `python3 src/<script>.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

Run: `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt` (on a PC; `.venv/` is already in `.gitignore`).

- [ ] **Step 2: Write the failing tests** — `tests/test_camera.py`

```python
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
```

- [ ] **Step 3: Run the tests to check they fail**

Run: `.venv/bin/python -m pytest tests/test_camera.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'camera'`.

- [ ] **Step 4: Write `src/camera.py`**

```python
"""Shared camera input for Jetson: CSI cameras (GStreamer), USB webcams (V4L2) and video files."""
import os
import re

import cv2 as cv

NO_DISPLAY_MSG = "No display found ($DISPLAY is not set). Run with --headless, or use X forwarding / a monitor."


class CameraError(RuntimeError):
    """Raised when a camera source cannot be opened."""


def gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30, flip=0):
    # nvarguscamerasrc runs the ISP (debayer, auto exposure, white balance) on CSI cameras;
    # nvvidconv copies the frame out of NVMM memory so OpenCV can read it via appsink.
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )


def has_gstreamer():
    return re.search(r"GStreamer:\s*YES", cv.getBuildInformation()) is not None


def open_capture(source, *, sensor_id=0, device=0, width=1280, height=720, fps=30, flip=0):
    """Open 'csi', 'usb' or a video file path. Raises CameraError if it cannot be opened."""
    if source == "csi":
        if not has_gstreamer():
            raise CameraError(
                "This OpenCV build has no GStreamer support, so CSI cameras cannot be opened.\n"
                "Use the OpenCV that ships with JetPack instead of 'pip install opencv-python'."
            )
        cap = cv.VideoCapture(gstreamer_pipeline(sensor_id, width, height, fps, flip), cv.CAP_GSTREAMER)
    elif source == "usb":
        cap = cv.VideoCapture(device, cv.CAP_V4L2)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv.CAP_PROP_FPS, fps)
    else:
        cap = cv.VideoCapture(source)

    if not cap.isOpened():
        raise CameraError(f"Failed to open camera source '{source}'.")
    return cap


def open_capture_from_args(args):
    return open_capture(
        args.source,
        sensor_id=args.sensor_id,
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip=args.flip,
    )


def add_camera_args(parser):
    parser.add_argument("--source", default="csi", help="'csi', 'usb', or a path to a video file (default: csi)")
    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor id (default: 0)")
    parser.add_argument("--device", type=int, default=0, help="USB camera index, /dev/videoN (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip", type=int, default=0, help="CSI flip method 0-7, e.g. 2 = rotate 180 (default: 0)")


def display_available():
    return bool(os.environ.get("DISPLAY"))
```

- [ ] **Step 5: Run the tests to check they pass**

Run: `.venv/bin/python -m pytest tests/test_camera.py -v`
Expected: 6 passed.

- [ ] **Step 6: Rewrite `src/camera_test.py` to use `camera.py`**

```python
"""Camera test for Jetson: open a CSI camera, USB webcam or video file and show frames with FPS.

Examples:
    python3 src/camera_test.py                     # CSI camera (e.g. Raspberry Pi Camera v2 / IMX219)
    python3 src/camera_test.py --source usb        # first USB webcam (/dev/video0)
    python3 src/camera_test.py --source video.mp4  # video file
    python3 src/camera_test.py --headless          # no window, print FPS only (e.g. over SSH)
"""
import argparse
import sys
import time

import cv2 as cv

from camera import NO_DISPLAY_MSG, CameraError, add_camera_args, display_available, open_capture_from_args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_camera_args(parser)
    parser.add_argument("--headless", action="store_true", help="do not open a window; print FPS instead")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.headless and not display_available():
        print(NO_DISPLAY_MSG)
        sys.exit(1)

    try:
        cap = open_capture_from_args(args)
    except CameraError as e:
        print(e)
        sys.exit(1)

    frames = 0
    fps = 0.0
    t_last = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame (end of stream or camera disconnected).")
                break

            frames += 1
            now = time.perf_counter()
            if now - t_last >= 1.0:
                fps = frames / (now - t_last)
                frames = 0
                t_last = now
                if args.headless:
                    print(f"{fps:.1f} FPS ({frame.shape[1]}x{frame.shape[0]})")

            if not args.headless:
                cv.putText(frame, f"{fps:.1f} FPS", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow("Camera", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.headless:
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Smoke test `camera_test.py` on a generated video**

Run:
```bash
.venv/bin/python -c "import cv2,numpy as np; w=cv2.VideoWriter('/tmp/clip.avi',cv2.VideoWriter_fourcc(*'MJPG'),30,(320,240)); [w.write(np.full((240,320,3),i%255,np.uint8)) for i in range(3000)]; w.release()"
.venv/bin/python src/camera_test.py --source /tmp/clip.avi --headless; echo "exit=$?"
.venv/bin/python src/camera_test.py --headless; echo "exit=$?"
.venv/bin/python src/camera_test.py --source /tmp/nope.mp4 --headless; echo "exit=$?"
```
Expected: (1) at least one `... FPS (320x240)` line, then `Failed to grab frame ...`, `exit=0`; (2) the no-GStreamer message, `exit=1`; (3) `Failed to open camera source '/tmp/nope.mp4'.`, `exit=1`. None of them print a traceback.

- [ ] **Step 8: Commit**

```bash
git add src/camera.py src/camera_test.py tests/conftest.py tests/test_camera.py requirements-dev.txt
git commit -m "Extract shared camera module; add unit test setup"
```

---

### Task 3: Rolling stage timer

**Files:**
- Create: `src/timing.py`, `tests/test_timing.py`

**Interfaces:**
- Produces (used by Task 5):
  - `class StageTimer(window: int = 30)`
    - `.add(stage: str, ms: float) -> None`
    - `.measure(stage: str)`: context manager that records elapsed ms under `stage`
    - `.tick(now: float | None = None) -> None`: marks the end of a frame; `now` defaults to `time.perf_counter()`
    - `.averages() -> dict[str, float]`: mean ms per stage over the last `window` samples, in the order stages were first added
    - `.fps() -> float`: frames per second over the last `window` frame intervals; `0.0` until there are 2 ticks

- [ ] **Step 1: Write the failing tests** — `tests/test_timing.py`

```python
import pytest

from timing import StageTimer


def test_averages_keep_first_seen_order():
    t = StageTimer()
    t.add("capture", 2.0)
    t.add("inference", 10.0)
    t.add("capture", 4.0)
    assert list(t.averages()) == ["capture", "inference"]
    assert t.averages() == {"capture": 3.0, "inference": 10.0}


def test_averages_are_rolling():
    t = StageTimer(window=3)
    for ms in [100.0, 1.0, 2.0, 3.0]:
        t.add("draw", ms)
    assert t.averages()["draw"] == pytest.approx(2.0)


def test_fps_needs_two_ticks():
    t = StageTimer()
    assert t.fps() == 0.0
    t.tick(now=1.0)
    assert t.fps() == 0.0


def test_fps_from_ticks():
    t = StageTimer()
    for i in range(4):
        t.tick(now=i * 0.1)
    assert t.fps() == pytest.approx(10.0)


def test_fps_is_rolling():
    t = StageTimer(window=2)
    for now in [0.0, 5.0, 5.1, 5.2]:  # the slow first frame drops out of the window
        t.tick(now=now)
    assert t.fps() == pytest.approx(10.0)


def test_fps_zero_elapsed():
    t = StageTimer()
    t.tick(now=1.0)
    t.tick(now=1.0)
    assert t.fps() == 0.0


def test_measure_records_elapsed_ms():
    t = StageTimer()
    with t.measure("capture"):
        pass
    assert t.averages()["capture"] >= 0.0


def test_measure_records_even_on_exception():
    t = StageTimer()
    with pytest.raises(ValueError):
        with t.measure("capture"):
            raise ValueError
    assert "capture" in t.averages()
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `.venv/bin/python -m pytest tests/test_timing.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'timing'`.

- [ ] **Step 3: Write `src/timing.py`**

```python
"""Rolling per-stage latency and frame-rate measurement."""
import time
from collections import deque
from contextlib import contextmanager


class StageTimer:
    """Rolling average of per-stage latencies (ms) and FPS over the last `window` frames."""

    def __init__(self, window=30):
        self.window = window
        self._stages = {}
        self._frame_times = deque(maxlen=window + 1)  # window+1 timestamps = window intervals

    def add(self, stage, ms):
        if stage not in self._stages:
            self._stages[stage] = deque(maxlen=self.window)
        self._stages[stage].append(ms)

    @contextmanager
    def measure(self, stage):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, (time.perf_counter() - start) * 1e3)

    def tick(self, now=None):
        self._frame_times.append(time.perf_counter() if now is None else now)

    def averages(self):
        return {stage: sum(samples) / len(samples) for stage, samples in self._stages.items()}

    def fps(self):
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `.venv/bin/python -m pytest tests/test_timing.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/timing.py tests/test_timing.py
git commit -m "Add rolling stage timer"
```

---

### Task 4: Engine export script

**Files:**
- Create: `src/export.py`, `tests/test_export.py`

**Interfaces:**
- Produces (used by Task 5):
  - `MODELS_DIR: Path`: `<repo>/models`
  - `engine_path(model: str, precision: str, imgsz: int, models_dir: Path | None = None) -> Path`, where `models_dir=None` means the current `MODELS_DIR`
  - `parse_engine_name(path) -> tuple[str, str, int] | None`: `(model, precision, imgsz)`, or `None` if the name doesn't follow the convention
  - `export_command(model: str, precision: str, imgsz: int) -> str`: the exact CLI to build that engine
  - `export_kwargs(precision: str, imgsz: int, calib_data: str | None, quantize_api: bool) -> dict`
  - `parse_args(argv=None) -> argparse.Namespace`, `main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests** — `tests/test_export.py`

```python
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
])
def test_parse_args_rejects(argv):
    with pytest.raises(SystemExit):
        export.parse_args(argv)


def test_main_skips_existing_engine(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(export, "MODELS_DIR", tmp_path)
    (tmp_path / "yolo11n_fp16_640.engine").write_bytes(b"x")
    assert export.main(["--model", "yolo11n"]) == 0  # would fail importing ultralytics if it tried to build
    assert "already exists" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `.venv/bin/python -m pytest tests/test_export.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'export'`.

- [ ] **Step 3: Write `src/export.py`**

```python
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
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `.venv/bin/python -m pytest tests/test_export.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add src/export.py tests/test_export.py
git commit -m "Add TensorRT engine export script"
```

---

### Task 5: Detection demo

**Files:**
- Create: `src/detect.py`, `tests/test_detect.py`

**Interfaces:**
- Consumes: from `camera`: `NO_DISPLAY_MSG, CameraError, add_camera_args, display_available, open_capture_from_args`; from `export`: `export_command, parse_engine_name`; from `timing`: `StageTimer`.
- Produces:
  - `parse_args(argv=None)`, `main(argv=None) -> int`
  - `resolve_classes(spec: str | None, model_names: dict[int, str]) -> list[int] | None`, which raises `ValueError` listing the valid names
  - `source_fps(reported: float) -> float`
  - `open_writer(path: Path, frame_size: tuple[int, int], fps: float) -> cv.VideoWriter`, which raises `OSError`
  - `overlay_lines(timer: StageTimer) -> list[str]`
  - `missing_engine_message(engine: Path) -> str`
  - `load_model(engine: Path)`, which raises `RuntimeError`

- [ ] **Step 1: Write the failing tests** — `tests/test_detect.py`

```python
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


def test_main_missing_engine_exits_1(tmp_path, capsys):
    rc = detect.main(["--engine", str(tmp_path / "yolo11n_fp16_640.engine"), "--headless"])
    assert rc == 1
    assert "python3 src/export.py --model yolo11n --precision fp16 --imgsz 640" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to check they fail**

Run: `.venv/bin/python -m pytest tests/test_detect.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'detect'`.

- [ ] **Step 3: Write `src/detect.py`**

```python
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

from camera import NO_DISPLAY_MSG, CameraError, add_camera_args, display_available, open_capture_from_args
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

    if not args.headless and not display_available():
        print(NO_DISPLAY_MSG)
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
                    print(" | ".join(overlay_lines(timer)))
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
```

- [ ] **Step 4: Run the tests to check they pass**

Run: `.venv/bin/python -m pytest tests/test_detect.py -v`
Expected: 18 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests -v`
Expected: 50 passed (6 camera + 8 timing + 18 export + 18 detect).

- [ ] **Step 6: Commit**

```bash
git add src/detect.py tests/test_detect.py
git commit -m "Add live TensorRT detection demo"
```

---

### Task 6: Hardware checks + README  **[needs Jetson]**

**Files:**
- Modify: `README.md` (add a Detection section after "Quick Start"; add an AGPL note in "License")

**Interfaces:**
- Consumes: `docker/run.sh` (Task 1), `src/export.py` (Task 4), `src/detect.py` (Task 5).

- [ ] **Step 1: [Jetson] Build engines**

```bash
docker/run.sh python3 src/export.py --model yolo11n --precision fp16
docker/run.sh python3 src/export.py --model yolo11n --precision int8
docker/run.sh python3 src/export.py --model yolo11n --precision fp16   # second run: "already exists"
```
Expected: `models/yolo11n_fp16_640.engine` and `models/yolo11n_int8_640.engine` exist; the INT8 build prints the coco8 warning; the third command skips the build. If `--precision int8` fails because of an unexpected argument, check which of `int8`/`quantize` this Ultralytics version accepts and fix `export_kwargs`.

- [ ] **Step 2: [Jetson] Run the demo with each source**

```bash
docker/run.sh python3 src/detect.py --engine models/yolo11n_fp16_640.engine                     # CSI, window
docker/run.sh python3 src/detect.py --engine models/yolo11n_fp16_640.engine --source usb --classes person
docker/run.sh python3 src/detect.py --engine models/yolo11n_int8_640.engine --headless          # Ctrl+C after ~10 s
docker/run.sh python3 src/detect.py --engine models/yolo11n_fp16_640.engine --classes dog,unicorn
docker/run.sh python3 src/detect.py --engine models/missing_fp16_640.engine
```
Expected: boxes and an overlay with FPS plus capture/preprocess/inference/postprocess/draw ms; `--classes person` shows only people; headless prints one line per second; `unicorn` prints the list of valid names and exits 1; the missing engine prints the export command and exits 1. No tracebacks. (Skip the CSI line if Task 1's gate failed and no fallback was chosen yet.)

- [ ] **Step 3: [Jetson] Save a video and check it plays**

```bash
docker/run.sh python3 src/detect.py --engine models/yolo11n_fp16_640.engine --save models/demo/out.mp4 --headless
# Ctrl+C after ~10 s, then:
python3 -c "import cv2; c=cv2.VideoCapture('models/demo/out.mp4'); print(int(c.get(cv2.CAP_PROP_FRAME_COUNT)), 'frames')"
```
Expected: a frame count > 0 (the file is playable even after Ctrl+C).

- [ ] **Step 4: [Jetson] Measure FPS**

For each engine, run on the CSI source (or USB if CSI is unavailable) with `--headless` for ~30 s, and note the steady FPS and inference ms. Also note the power mode (`sudo nvpmodel -q`) and whether `sudo jetson_clocks` was on.

- [ ] **Step 5: Update the README**

Insert after the Quick Start section, replacing each value in the table with the numbers measured in Step 4:

````markdown
## Object Detection (TensorRT)

The detection demo runs [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models as TensorRT engines inside Ultralytics' Jetson Docker image, so you don't need to install PyTorch yourself.

**1. Build an engine** (once per device; engines only run on the device they were built on):
```bash
docker/run.sh python3 src/export.py --model yolo11n --precision fp16   # -> models/yolo11n_fp16_640.engine
docker/run.sh python3 src/export.py --model yolo11n --precision int8   # optional, faster, slightly less accurate
```

**2. Run detection:**
```bash
xhost +local:docker   # allow the container to open windows (once per login)
docker/run.sh python3 src/detect.py --engine models/yolo11n_fp16_640.engine
```

Useful options (all of `camera_test.py`'s camera options work too):

| Option | Meaning |
|---|---|
| `--source csi\|usb\|<file>` | Camera or video file (default: `csi`) |
| `--classes person,car` | Only detect these classes |
| `--conf 0.5` | Confidence threshold, 0–1 (default: 0.25) |
| `--save out.mp4` | Save the annotated video |
| `--headless` | No window; print FPS and per-stage timings every second |

The on-screen overlay shows FPS and the average time for each stage: capture, preprocess, inference, postprocess and draw.

**Measured on Jetson AGX Orin** (yolo11n, 640×640, 1280×720 CSI input, power mode <MODE>, jetson_clocks <on/off>):

| Engine | FPS | Inference (ms) |
|---|---|---|
| FP16 | <FPS> | <MS> |
| INT8 | <FPS> | <MS> |
````

Change the License section to:
```markdown
## License
MIT License.

The detection demo uses [Ultralytics](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. If you build and distribute a product on top of it, AGPL terms apply to that product unless you have an Ultralytics Enterprise license.
```

Then check that no `<MODE>`, `<FPS>`, `<MS>` or `<on/off>` markers are left: `grep -nE '<(MODE|FPS|MS|on/off)>' README.md` should print nothing.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Document TensorRT detection demo with measured FPS"
```
