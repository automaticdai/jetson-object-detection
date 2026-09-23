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
