# jetson-object-detection

Following the success of my other project, [rpi-object-detection](https://github.com/automaticdai/rpi-object-detection), I am making this new project focusing on Nvidia Jetson. As many configurations are different on Jetson, I think this is a better way to have a new repo, rather than a hybrid integration to support multiple platforms. Due to the on-board GPU, with Jetson, I am going to produce more interesting CV-based projects.

Star the project and stay tuned!

## Hardware
This repo is developed with Jetson AGX Orin, and it should support the following hardware:

- Jetson AGX Thor (2025)
- Jetson Orin Nano Super (2024)
- Jetson AGX Orin (2023)
- Jetson Orin NX (2023)
- Jetson Orin Nano (2023)

However, there is no support for older boards, including:

- Jetson Xavier NX (2020)
- Jetson Nano (2019)
- Jetson AGX Xavier (2018)
- Jetson TX2 (2017)
- Jetson TX1 (2015)
- Jetson TK1 (2014)

Note that this does not mean the code won't be able to run on these boards --- I just don't have the time and energy to test them.

## Requirements
- **JetPack 6.x** (Jetson Linux 36.x) for Orin boards, or **JetPack 7.x** for Jetson Thor. Check your version with `cat /etc/nv_tegra_release` or `apt show nvidia-jetpack`.
- **Python 3.10+** (the system Python that ships with JetPack).
- **OpenCV 4.x with GStreamer support**. JetPack's OpenCV already has this. Do **not** `pip install opencv-python`: the pip wheels are built without GStreamer, so CSI cameras won't open. Check with:
  ```bash
  python3 -c "import cv2; print(cv2.__version__); print([l for l in cv2.getBuildInformation().splitlines() if 'GStreamer' in l])"
  ```
- **PyTorch built for Jetson** (only needed for the upcoming detection demos). The default `pip install torch` wheels have no CUDA support on Jetson. Use NVIDIA's [PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) wheels, or a [jetson-containers](https://github.com/dusty-nv/jetson-containers) image.
- **A camera**: a CSI camera (e.g. Raspberry Pi Camera Module v2 / IMX219) or a USB webcam.

## Quick Start
```bash
git clone https://github.com/automaticdai/jetson-object-detection.git
cd jetson-object-detection

# CSI camera (default). Press 'q' to quit.
python3 src/camera_test.py

# USB webcam
python3 src/camera_test.py --source usb

# Over SSH / no monitor: print FPS instead of opening a window
python3 src/camera_test.py --headless

# All options
python3 src/camera_test.py --help
```

If the CSI camera fails to open, check it is detected (`ls /dev/video*`), then restart the camera daemon with `sudo systemctl restart nvargus-daemon`.

## License
MIT License
