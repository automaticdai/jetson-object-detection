# TensorRT Object Detection Demo — Design

Date: 2026-09-23
Status: Draft, awaiting review

## Goal

Add the repo's first real object-detection feature: a live YOLO demo running on TensorRT on Jetson Orin / Thor.

- **Audience:** learners and hobbyists first (clone → live detection in minutes), while laying groundwork for a later benchmarking tool.
- **Success criteria:**
  1. A user with a supported Jetson and a camera can go from clone to live detection on screen by following the README.
  2. The demo reports FPS and a per-stage latency breakdown (capture / preprocess / inference / postprocess / draw).
  3. FP16 and INT8 engines can be built and compared on the same board.

## Decisions

| Topic | Decision | Reason |
|---|---|---|
| Inference stack | Ultralytics (`export(format="engine")`, `YOLO("x.engine")`) | Fewest lines of code; handles pre/post-processing and model swaps. |
| Licence | Ultralytics is AGPL-3.0; repo code stays MIT. README states this. | Using it as a dependency is fine for a demo; downstream products inherit AGPL. |
| Environment | Docker, based on Ultralytics' official Jetson image | Avoids installing NVIDIA's PyTorch wheels by hand, which is the most fragile step. |
| Threading | Single-threaded loop | Honest per-stage timing; the breakdown is the base of the benchmark phase. |
| Engine build | Separate `export.py` step, not automatic on first run | Export takes minutes and depends on the board; it should be explicit. |

Out of scope for this spec: multi-threaded capture, RTSP output, tracking, raw TensorRT backend, the full benchmark tool (a later spec).

## Components

```
src/
  camera.py       # NEW: gstreamer_pipeline(), open_capture(); moved out of camera_test.py
  camera_test.py  # imports camera.py; behaviour unchanged
  export.py       # NEW: builds TensorRT engines
  detect.py       # NEW: live detection demo
docker/
  run.sh          # NEW: runs the Ultralytics Jetson container with camera + display access
models/           # built engines (git-ignored)
tests/            # NEW: pytest unit tests (no GPU)
.gitignore        # add models/, *.pt, *.onnx, *.engine, *.mp4
```

### `src/camera.py`
- `gstreamer_pipeline(sensor_id, width, height, fps, flip) -> str`
- `open_capture(source, *, sensor_id, device, width, height, fps, flip) -> cv.VideoCapture`
  - `source`: `"csi"`, `"usb"`, or a file path. Same behaviour as the current `camera_test.py`, including the "no GStreamer" error.
- `add_camera_args(parser)`: adds the shared `--source/--sensor-id/--device/--width/--height/--fps/--flip` options, so both scripts accept the same camera flags.

### `src/export.py`
```
python3 src/export.py --model yolo11n --precision fp16|int8 --imgsz 640 [--calib-data coco8.yaml] [--force]
```
- Output: `models/<model>_<precision>_<imgsz>.engine` (e.g. `models/yolo11n_fp16_640.engine`). The name is built by one function, `engine_path(model, precision, imgsz)`, which is unit-tested.
- Skips the build if the engine already exists, unless `--force` is given.
- INT8: `export(format="engine", int8=True, data=<calib-data>)` on older Ultralytics, `quantize=8` on newer versions. The script checks which one the installed version supports (whether `quantize` is in `ultralytics.utils.DEFAULT_CFG_DICT`). If `--calib-data` is left out, it uses `coco8.yaml` and warns that such a small calibration set may reduce accuracy.
- FP16: `export(format="engine", half=True)`.
- Downloads the `.pt` weights into `models/` if they're missing (Ultralytics does this automatically).
- Engines only run on the board/TensorRT version they were built with. The README and `--help` say to export on the target device.

### `src/detect.py`
```
python3 src/detect.py --engine models/yolo11n_fp16_640.engine [camera flags] \
    [--conf 0.25] [--classes person,car] [--save out.mp4] [--headless]
```
- Loads `YOLO(engine, task="detect")` once, then runs one warm-up `predict` on a blank frame before the loop.
- `--classes`: names are converted to ids through `model.names`. An unknown name is an error that lists the valid names.
- `--save`: a `cv.VideoWriter` (mp4v) is created lazily on the first frame, using the frame's size and the source FPS (30 if unknown).
- `--headless`: no window; prints FPS and stage timings once per second.

### `docker/run.sh`
- Chooses the image tag from the JetPack version (`/etc/nv_tegra_release`), with a `--image` override.
- `docker run --rm -it --runtime nvidia --ipc=host`
  - `-v <repo>:/app -w /app`
  - `-v /tmp/argus_socket:/tmp/argus_socket` (CSI camera daemon)
  - `--device /dev/videoN` for each existing video device
  - `-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`, with a reminder to run `xhost +local:docker`
- Any arguments are passed through as the command to run, e.g. `docker/run.sh python3 src/detect.py ...`.

## Data flow (`detect.py`)

```
open_capture ─► read frame ─► model.predict(frame, conf, classes, verbose=False)
                                  │  result.speed → preprocess / inference / postprocess ms
                  draw boxes (result.plot()) + overlay (FPS, per-stage ms)
                                  ├─► cv.imshow       (unless --headless)
                                  └─► VideoWriter     (if --save)
```

- `imgsz` is not passed to `predict`: a TensorRT engine has a fixed input size, and Ultralytics reads it from the engine's metadata.
- Capture and draw are timed with `time.perf_counter()`. The pre/infer/post times come from `result.speed`.
- `StageTimer`: a rolling average over the last 30 frames for each stage, plus FPS. It's a pure-Python class and unit-tested.

## Error handling

Every case below prints a clear message and exits with a non-zero code:

| Case | Behaviour |
|---|---|
| Engine file missing | Print the exact `export.py` command that builds it. |
| Engine fails to load (built on another board/TensorRT version) | Catch the load error; tell the user to re-export on this device. |
| Unknown class name | List the valid class names. |
| Camera errors | Reuse `camera.py` messages (no GStreamer, cannot open source, no display). |
| INT8 without `--calib-data` | Warn, then fall back to `coco8.yaml`. |
| Ctrl+C / end of stream | `try/finally` releases the capture and the writer, so partial videos are still playable. |

## Testing

**Unit tests (pytest, no GPU, suitable for CI).** Heavy imports (`ultralytics`) are done inside functions, so these tests don't need them installed:
- `engine_path()` naming
- class-name → id conversion, including the unknown-name error
- `StageTimer` rolling averages and FPS
- `gstreamer_pipeline()` string
- argument parsing for `detect.py` and `export.py`

**Hardware checklist (manual, on the AGX Orin):**
1. **Risk check first:** inside the Docker image, `python3 src/camera_test.py --source csi` works. If the image's OpenCV has no GStreamer, stop and choose a fallback (see Risks).
2. `export.py` builds FP16 and INT8 engines for yolo11n.
3. `detect.py` works with a CSI camera, a USB camera, and a video file.
4. `--save` produces a playable MP4; `--headless` prints timings.
5. Record FP16 vs INT8 FPS for yolo11n on the AGX Orin in the README.

## Risks

1. **CSI inside Docker.** The Ultralytics Jetson image probably installs OpenCV from pip, which has no GStreamer, so CSI cameras won't open. USB and file sources still work.
   *Fallback:* a small custom image `FROM` the Ultralytics image, with a GStreamer-enabled OpenCV (from JetPack's apt packages, or built from source). This is decided after hardware check 1.
2. **Thor / JetPack 7 image.** Ultralytics publishes `ultralytics/ultralytics:latest-nvidia-arm64` for JetPack 7.0 (Thor). Its docs say JetPack 7.2 needs a native install for now. `run.sh` maps L4T R36 → `latest-jetson-jetpack6` and R38 → `latest-nvidia-arm64`. Thor stays "untested" until someone runs it.
3. **Ultralytics API drift.** Pin the image tag in `run.sh`, and note the Ultralytics version the demo was tested with.
