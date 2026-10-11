from abc import ABC, abstractmethod

from cs336_systems.benchmark.dto import BenchmarkResult, BenchmarkType
from cs336_systems.benchmark.workload import Workload


class ProfilerStep(ABC):
    @abstractmethod
    def __call__(self, workload: Workload, benchmark_type: BenchmarkType) -> BenchmarkResult | None:
        pass
