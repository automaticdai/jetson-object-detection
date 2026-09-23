"""Camera test for Jetson: open a CSI camera, USB webcam or video file and show frames with FPS.

Examples:
    python3 src/camera_test.py                     # CSI camera (e.g. Raspberry Pi Camera v2 / IMX219)
    python3 src/camera_test.py --source usb        # first USB webcam (/dev/video0)
    python3 src/camera_test.py --source video.mp4  # video file
    python3 src/camera_test.py --headless          # no window, print FPS only (e.g. over SSH)
"""
import argparse
import os
import re
import sys
import time

import cv2 as cv


def gstreamer_pipeline(sensor_id=0, width=1280, height=720, fps=30, flip=0):
    # nvarguscamerasrc runs the ISP (debayer, auto exposure, white balance) on CSI cameras;
    # nvvidconv copies the frame out of NVMM memory so OpenCV can read it via appsink.
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )


def open_capture(args):
    if args.source == "csi":
        if not re.search(r"GStreamer:\s*YES", cv.getBuildInformation()):
            print("This OpenCV build has no GStreamer support, so CSI cameras cannot be opened.\n"
                  "Use the OpenCV that ships with JetPack instead of 'pip install opencv-python'.")
            sys.exit(1)
        pipeline = gstreamer_pipeline(args.sensor_id, args.width, args.height, args.fps, args.flip)
        return cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    if args.source == "usb":
        cap = cv.VideoCapture(args.device, cv.CAP_V4L2)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv.CAP_PROP_FPS, args.fps)
        return cap
    return cv.VideoCapture(args.source)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="csi", help="'csi', 'usb', or a path to a video file (default: csi)")
    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor id (default: 0)")
    parser.add_argument("--device", type=int, default=0, help="USB camera index, /dev/videoN (default: 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip", type=int, default=0, help="CSI flip method 0-7, e.g. 2 = rotate 180 (default: 0)")
    parser.add_argument("--headless", action="store_true", help="do not open a window; print FPS instead")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.headless and not os.environ.get("DISPLAY"):
        print("No display found ($DISPLAY is not set). Run with --headless, or use X forwarding / a monitor.")
        sys.exit(1)

    cap = open_capture(args)
    if not cap.isOpened():
        print(f"Failed to open camera source '{args.source}'.")
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
