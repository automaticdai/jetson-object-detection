"""Rolling per-stage latency and frame-rate measurement."""
import time
from collections import deque
from contextlib import contextmanager


class StageTimer:
    """Rolling average of per-stage latencies (ms) and FPS over the last `window` frames."""

    def __init__(self, window=30):
        self.window = window
        self._stages = {}
        self._frame_times = deque(maxlen=window + 1)  # window+1 timestamps = window intervals

    def add(self, stage, ms):
        if stage not in self._stages:
            self._stages[stage] = deque(maxlen=self.window)
        self._stages[stage].append(ms)

    @contextmanager
    def measure(self, stage):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, (time.perf_counter() - start) * 1e3)

    def tick(self, now=None):
        self._frame_times.append(time.perf_counter() if now is None else now)

    def averages(self):
        return {stage: sum(samples) / len(samples) for stage, samples in self._stages.items()}

    def fps(self):
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0
