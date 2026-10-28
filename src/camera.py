"""Shared camera input for Jetson: CSI cameras (GStreamer), USB webcams (V4L2) and video files."""
import os
import queue
import re
import threading

import cv2 as cv

NO_DISPLAY_MSG = "No display found ($DISPLAY is not set). Run with --headless, or use X forwarding / a monitor."
NO_GUI_MSG = (
    "This OpenCV build has no GUI support, so it cannot open a window "
    "(the Ultralytics Docker image ships opencv-python-headless).\n"
    "Run with --headless (add --save out.mp4 to record the result)."
)


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
        # Most USB webcams only reach 30 FPS at 720p+ with MJPG; the default (YUYV) is often 10 FPS or less.
        # Cameras without MJPG ignore this and keep their default format.
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv.CAP_PROP_FPS, fps)
    else:
        cap = cv.VideoCapture(source)

    if not cap.isOpened():
        raise CameraError(f"Failed to open camera source '{source}'.")
    return cap


class ThreadedCapture:
    """Reads frames on a background thread, so capture overlaps with inference.

    drop=True (cameras): keep only the newest frame, so a slow consumer always gets a fresh one.
    drop=False (video files): queue frames, so none are skipped.
    Has the read() / get() / release() subset of cv.VideoCapture that the scripts use.
    """

    def __init__(self, cap, drop=True, queue_size=4):
        self._cap = cap
        self._drop = drop
        self._queue = queue.Queue(maxsize=1 if drop else queue_size)
        self._stopped = threading.Event()
        self._done = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        ret = False
        try:
            while not self._stopped.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    break
                if self._drop:
                    try:
                        self._queue.get_nowait()  # only this thread puts, so the slot is now free
                    except queue.Empty:
                        pass
                self._put((True, frame))
        finally:
            # Blocks until the consumer has taken the last frame, so it is never dropped.
            self._put((False, None))

    def _put(self, item):
        while not self._stopped.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def read(self):
        if self._done:
            return False, None
        ret, frame = self._queue.get()
        if not ret:
            self._done = True
        return ret, frame

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        self._stopped.set()
        self._thread.join(timeout=2)
        self._cap.release()


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


def has_gui():
    return re.search(r"^\s*GUI:\s*NONE\s*$", cv.getBuildInformation(), re.MULTILINE) is None


def window_error(headless):
    """Why a preview window cannot be opened, or None if it can (or none is needed)."""
    if headless:
        return None
    if not display_available():
        return NO_DISPLAY_MSG
    if not has_gui():
        return NO_GUI_MSG
    return None
