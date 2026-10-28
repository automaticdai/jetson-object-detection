import csv

import cv2 as cv
import numpy as np
import pytest

import benchmark


def test_percentile_nearest_rank():
    values = list(range(1, 101))  # 1..100
    assert benchmark.percentile(values, 50) == 50
    assert benchmark.percentile(values, 95) == 95
    assert benchmark.percentile([7.0], 95) == 7.0


def test_summarize():
    row = benchmark.summarize("m.engine", [10.0, 10.0, 20.0, 40.0], {"inference": [6.0, 8.0, 8.0, 10.0]})
    assert row["engine"] == "m.engine"
    assert row["fps"] == pytest.approx(4 / 0.08)
    assert row["inference"] == 8.0
    assert (row["p50_ms"], row["p95_ms"]) == (10.0, 40.0)


def test_format_table():
    row = {"engine": "a.engine", "fps": 123.45, "preprocess": 1.0, "inference": 5.55, "postprocess": 0.5,
           "p50_ms": 8.0, "p95_ms": 9.0}
    lines = benchmark.format_table([row]).splitlines()
    assert lines[0].startswith("| Engine | FPS |")
    assert lines[2] == "| a.engine | 123.5 | 1.0 | 5.5 | 0.5 | 8.0 | 9.0 |"


def test_load_frames_loops_a_short_video(tmp_path):
    path = tmp_path / "short.avi"
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for value in (0, 100, 200):
        writer.write(np.full((24, 32, 3), value, np.uint8))
    writer.release()
    frames = benchmark.load_frames(str(path), 7)
    assert len(frames) == 7
    assert [round(f.mean() / 100) for f in frames] == [0, 1, 2, 0, 1, 2, 0]


def test_load_frames_from_image(tmp_path):
    path = tmp_path / "img.png"
    cv.imwrite(str(path), np.zeros((24, 32, 3), np.uint8))
    assert len(benchmark.load_frames(str(path), 5)) == 5


def test_load_frames_unreadable_source(tmp_path):
    with pytest.raises(ValueError, match="nope.mp4"):
        benchmark.load_frames(str(tmp_path / "nope.mp4"), 5)


def test_default_engines_sorted(tmp_path):
    for name in ("b_fp16_640.engine", "a_int8_640.engine", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in benchmark.default_engines(tmp_path)] == ["a_int8_640.engine", "b_fp16_640.engine"]


@pytest.mark.parametrize("argv", [["--frames", "0"], ["--warmup", "-1"]])
def test_bad_counts_rejected(argv):
    with pytest.raises(SystemExit):
        benchmark.parse_args(argv)


def test_write_csv_adds_device_column(tmp_path):
    path = tmp_path / "out" / "r.csv"
    benchmark.write_csv(path, [{"engine": "a.engine", "fps": 10.0 / 3}], "Jetson Test")
    rows = list(csv.DictReader(open(path)))
    assert rows == [{"device": "Jetson Test", "engine": "a.engine", "fps": "3.33"}]


def test_main_without_engines(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(benchmark, "default_engines", lambda: [])
    assert benchmark.main([]) == 1
    assert "src/export.py" in capsys.readouterr().out


def test_main_missing_engine(tmp_path, capsys):
    assert benchmark.main(["--engines", str(tmp_path / "x.engine")]) == 1
    assert "x.engine" in capsys.readouterr().out
