import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


def synchronize_cuda() -> None:
    """Synchronize GPU work before wall-clock measurements."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timer_start() -> float:
    synchronize_cuda()
    return perf_counter()


def timer_stop(start: float) -> float:
    synchronize_cuda()
    return perf_counter() - start


class Benchmarker:
    def __init__(self):
        self.execution_times = defaultdict(list)

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        try:
            start_time = timer_start()
            yield
        finally:
            elapsed = timer_stop(start_time)
            for _ in range(num_calls):
                self.execution_times[tag].append(elapsed / num_calls)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.execution_times), f)

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(torch.cuda.memory_stats()["allocated_bytes.all.peak"], f)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")

    def clear_history(self) -> None:
        self.execution_times = defaultdict(list)
