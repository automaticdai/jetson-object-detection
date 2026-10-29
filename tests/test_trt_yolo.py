import numpy as np
import pytest

import trt_yolo


def test_coco_names():
    assert len(trt_yolo.NAMES) == 80
    assert (trt_yolo.NAMES[0], trt_yolo.NAMES[9], trt_yolo.NAMES[79]) == ("person", "traffic light", "toothbrush")


def test_letterbox_pads_to_square_and_centres():
    image = np.full((360, 640, 3), 255, np.uint8)  # 16:9
    padded, scale, (left, top) = trt_yolo.letterbox(image, 320)
    assert padded.shape == (320, 320, 3)
    assert scale == 0.5 and (left, top) == (0, 70)
    assert padded[0, 0].tolist() == [114, 114, 114] and padded[160, 160].tolist() == [255, 255, 255]


def test_preprocess_layout():
    image = np.zeros((48, 64, 3), np.uint8)
    image[..., 2] = 255  # red in BGR
    blob, _, _ = trt_yolo.preprocess(image, 64)
    assert blob.shape == (1, 3, 64, 64) and blob.dtype == np.float32 and blob.flags["C_CONTIGUOUS"]
    assert blob[0, 0, 32, 32] == 1.0 and blob[0, 2, 32, 32] == 0.0  # channel 0 is red (RGB)


def make_output(rows, nc=3, anchors=10):
    """rows: (cx, cy, w, h, class, score) in letterboxed pixels -> (1, 4 + nc, anchors) like YOLO's output."""
    out = np.zeros((1, 4 + nc, anchors), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


def test_postprocess_undoes_letterbox():
    # image 640x360 letterboxed to 320: scale 0.5, pad top 70
    out = make_output([(160, 160, 100, 50, 1, 0.9)])
    dets = trt_yolo.postprocess(out, 0.5, (0, 70), (360, 640, 3))
    assert dets.shape == (1, 6)
    np.testing.assert_allclose(dets[0], [220, 130, 420, 230, 0.9, 1], rtol=1e-5)


def test_postprocess_conf_nms_and_classes():
    out = make_output([
        (100, 100, 50, 50, 0, 0.9),
        (102, 101, 50, 50, 0, 0.8),   # overlaps the first, same class -> suppressed
        (102, 101, 50, 50, 2, 0.7),   # same place, other class -> kept
        (200, 200, 20, 20, 1, 0.1),   # below conf
    ])
    dets = trt_yolo.postprocess(out, 1.0, (0, 0), (320, 320, 3), conf=0.25, iou=0.45)
    assert sorted(dets[:, 5].tolist()) == [0, 2]
    only = trt_yolo.postprocess(out, 1.0, (0, 0), (320, 320, 3), conf=0.25, classes=[2])
    assert only[:, 5].tolist() == [2]


def test_postprocess_nothing_found():
    dets = trt_yolo.postprocess(make_output([]), 1.0, (0, 0), (320, 320, 3))
    assert dets.shape == (0, 6)


def test_postprocess_clips_to_image():
    out = make_output([(5, 5, 40, 40, 0, 0.9)])
    x1, y1, x2, y2 = trt_yolo.postprocess(out, 1.0, (0, 0), (100, 100, 3))[0, :4]
    assert (x1, y1, x2, y2) == (0, 0, 25, 25)


def test_result_plot_draws_on_a_copy():
    image = np.zeros((100, 100, 3), np.uint8)
    dets = np.array([[10, 10, 60, 60, 0.9, 0]], np.float32)
    result = trt_yolo.Result(image, dets, trt_yolo.NAMES, {"inference": 1.0})
    plotted = result.plot()
    assert plotted.any() and not image.any()
