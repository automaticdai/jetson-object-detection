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
