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
    # `|| true`: without it, a missing file would abort the script silently under `set -euo pipefail`.
    release="$(head -n1 /etc/nv_tegra_release 2>/dev/null | sed -n 's/^# R\([0-9]*\).*/\1/p' || true)"
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
