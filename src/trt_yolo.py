"""Run a YOLO detection engine with TensorRT directly, without Ultralytics or PyTorch.

For boards whose JetPack is too old for Ultralytics, e.g. Jetson Nano (JetPack 4, Python 3.6, TensorRT 8.2).
Needs only the `tensorrt` Python bindings that ship with JetPack, NumPy and OpenCV; CUDA memory is
managed through libcudart with ctypes, so pycuda is not needed.

The engine is built on the Jetson from an ONNX export of a YOLOv8/YOLO11 detection model, whose output
is (1, 4 + classes, anchors): box centre/size, then one score per class.

Mimics the small part of the Ultralytics API that detect.py and benchmark.py use:
model.names, model.predict(...)[0].speed and .plot().
"""
import ctypes
import time

import cv2 as cv
import numpy as np

COCO_NAMES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
    "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
    "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
    "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
    "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush"
)
NAMES = {i: name.replace("_", " ") for i, name in enumerate(COCO_NAMES.split())}

_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2


def letterbox(image, size, color=(114, 114, 114)):
    """Resize keeping the aspect ratio and pad to size x size, as Ultralytics does.

    Returns the padded image, the scale factor and the (left, top) padding.
    """
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    padded = cv.copyMakeBorder(
        resized, top, size - new_h - top, left, size - new_w - left, cv.BORDER_CONSTANT, value=color
    )
    return padded, scale, (left, top)


def preprocess(image, size):
    """BGR frame -> (1, 3, size, size) float32 RGB tensor in [0, 1], plus what's needed to undo the letterbox."""
    padded, scale, pad = letterbox(image, size)
    blob = padded[:, :, ::-1].transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), scale, pad


def postprocess(output, scale, pad, image_shape, conf=0.25, iou=0.45, classes=None):
    """(1, 4 + nc, anchors) raw output -> (N, 6) array of x1, y1, x2, y2, score, class in image pixels."""
    preds = output[0].T  # (anchors, 4 + nc)
    scores_all = preds[:, 4:]
    class_ids = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(preds)), class_ids]
    keep = scores >= conf
    if classes is not None:
        keep &= np.isin(class_ids, classes)
    boxes, scores, class_ids = preds[keep, :4], scores[keep], class_ids[keep]
    if len(boxes) == 0:
        return np.zeros((0, 6), np.float32)

    # centre/size in letterboxed pixels -> corners in image pixels
    xy = (boxes[:, :2] - np.array(pad, np.float32)) / scale
    wh = boxes[:, 2:4] / scale
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    h, w = image_shape[:2]
    x1y1 = np.clip(x1y1, 0, [w, h])
    x2y2 = np.clip(x2y2, 0, [w, h])

    # Class-aware NMS in one call: shift each class far apart so boxes of different classes never overlap.
    offset = class_ids[:, None].astype(np.float32) * 4096
    nms_boxes = np.hstack([x1y1 + offset, x2y2 - x1y1]).tolist()
    idx = np.array(cv.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf, iou), dtype=int).reshape(-1)
    return np.hstack([x1y1, x2y2, scores[:, None], class_ids[:, None]])[idx].astype(np.float32)


def class_color(class_id):
    hue = int(class_id) * 47 % 180
    bgr = cv.cvtColor(np.uint8([[[hue, 200, 255]]]), cv.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def draw_detections(image, detections, names):
    for x1, y1, x2, y2, score, class_id in detections:
        color = class_color(class_id)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv.rectangle(image, p1, p2, color, 2)
        label = "{} {:.2f}".format(names.get(int(class_id), int(class_id)), score)
        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(p1[1], th + 6)
        cv.rectangle(image, (p1[0], top - th - 6), (p1[0] + tw + 4, top), color, -1)
        cv.putText(image, label, (p1[0] + 2, top - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


class Result:
    def __init__(self, image, detections, names, speed):
        self.orig_img = image
        self.detections = detections
        self.names = names
        self.speed = speed  # ms per stage, like Ultralytics' Results.speed

    def plot(self):
        return draw_detections(self.orig_img.copy(), self.detections, self.names)


class _CudaRuntime:
    def __init__(self):
        for name in ("libcudart.so", "/usr/local/cuda/lib64/libcudart.so"):
            try:
                self._lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("Cannot load libcudart.so. Is CUDA installed (it ships with JetPack)?")
        self._lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._lib.cudaFree.argtypes = [ctypes.c_void_p]

    def _check(self, status, what):
        if status != 0:
            raise RuntimeError("{} failed with CUDA error {}".format(what, status))

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self._lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr.value

    def to_device(self, dst, array):
        self._check(self._lib.cudaMemcpy(dst, array.ctypes.data, array.nbytes, _CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy")

    def to_host(self, array, src):
        self._check(self._lib.cudaMemcpy(array.ctypes.data, src, array.nbytes, _CUDA_MEMCPY_DEVICE_TO_HOST), "cudaMemcpy")

    def free(self, ptr):
        self._lib.cudaFree(ptr)


class TrtYolo:
    """A YOLO detection engine run with TensorRT. Use .predict(frame) like an Ultralytics model."""

    def __init__(self, engine_path, names=None, iou=0.45):
        import tensorrt as trt

        self.names = names or NAMES
        self.iou = iou
        self._cuda = _CudaRuntime()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(str(engine_path), "rb") as f, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(
                "TensorRT could not load {}. Engines only run on the device and TensorRT version "
                "they were built with; rebuild it on this device with trtexec.".format(engine_path)
            )
        self._context = self._engine.create_execution_context()

        self._bindings, self._device_ptrs = [], []
        self._input = self._output = None
        for i in range(self._engine.num_bindings):
            shape = tuple(self._engine.get_binding_shape(i))
            host = np.empty(shape, dtype=trt.nptype(self._engine.get_binding_dtype(i)))
            ptr = self._cuda.malloc(host.nbytes)
            self._device_ptrs.append(ptr)
            self._bindings.append(ptr)
            if self._engine.binding_is_input(i):
                self._input = (host, ptr)
            else:
                self._output = (host, ptr)
        self.imgsz = self._input[0].shape[-1]
        nc = self._output[0].shape[1] - 4
        if nc != len(self.names):
            self.names = {i: str(i) for i in range(nc)}  # custom model: number its classes

        self.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8))  # warm up, so the first real frame is honest

    def infer(self, blob):
        host_in, ptr_in = self._input
        host_out, ptr_out = self._output
        self._cuda.to_device(ptr_in, blob.astype(host_in.dtype, copy=False))
        self._context.execute_v2(self._bindings)  # synchronous
        self._cuda.to_host(host_out, ptr_out)
        return host_out.astype(np.float32, copy=False)

    def predict(self, frame, conf=0.25, classes=None, verbose=False):
        t0 = time.perf_counter()
        blob, scale, pad = preprocess(frame, self.imgsz)
        t1 = time.perf_counter()
        output = self.infer(blob)
        t2 = time.perf_counter()
        detections = postprocess(output, scale, pad, frame.shape, conf, self.iou, classes)
        t3 = time.perf_counter()
        speed = {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}
        return [Result(frame, detections, self.names, speed)]

    def __del__(self):
        for ptr in getattr(self, "_device_ptrs", []):
            self._cuda.free(ptr)
