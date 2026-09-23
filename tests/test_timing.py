import pytest

from timing import StageTimer


def test_averages_keep_first_seen_order():
    t = StageTimer()
    t.add("capture", 2.0)
    t.add("inference", 10.0)
    t.add("capture", 4.0)
    assert list(t.averages()) == ["capture", "inference"]
    assert t.averages() == {"capture": 3.0, "inference": 10.0}


def test_averages_are_rolling():
    t = StageTimer(window=3)
    for ms in [100.0, 1.0, 2.0, 3.0]:
        t.add("draw", ms)
    assert t.averages()["draw"] == pytest.approx(2.0)


def test_fps_needs_two_ticks():
    t = StageTimer()
    assert t.fps() == 0.0
    t.tick(now=1.0)
    assert t.fps() == 0.0


def test_fps_from_ticks():
    t = StageTimer()
    for i in range(4):
        t.tick(now=i * 0.1)
    assert t.fps() == pytest.approx(10.0)


def test_fps_is_rolling():
    t = StageTimer(window=2)
    for now in [0.0, 5.0, 5.1, 5.2]:  # the slow first frame drops out of the window
        t.tick(now=now)
    assert t.fps() == pytest.approx(10.0)


def test_fps_zero_elapsed():
    t = StageTimer()
    t.tick(now=1.0)
    t.tick(now=1.0)
    assert t.fps() == 0.0


def test_measure_records_elapsed_ms():
    t = StageTimer()
    with t.measure("capture"):
        pass
    assert t.averages()["capture"] >= 0.0


def test_measure_records_even_on_exception():
    t = StageTimer()
    with pytest.raises(ValueError):
        with t.measure("capture"):
            raise ValueError
    assert "capture" in t.averages()
