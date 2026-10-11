import timeit

import torch

from cs336_systems.benchmark.dto import BenchmarkResult, BenchmarkType
from cs336_systems.benchmark.runner.step import ProfilerStep
from cs336_systems.benchmark.workload import Workload


class TimeManager:
    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self.elapsed: float = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = timeit.default_timer() - self.start_time


class TimingStep(ProfilerStep):
    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = device

    def __call__(self, workload: Workload, benchmark_type: BenchmarkType) -> BenchmarkResult:
        result = BenchmarkResult()
        with TimeManager(self.device) as t:
            workload.forward()
        result.forward = t.elapsed

        if benchmark_type != BenchmarkType.FORWARD:
            with TimeManager(self.device) as t:
                workload.backward()
            result.backward = t.elapsed

            if benchmark_type == BenchmarkType.FORWARD_BACKWARD_OPTIMIZER:
                with TimeManager(self.device) as t:
                    workload.optimizer_step()
                result.optimizer = t.elapsed

        return result
