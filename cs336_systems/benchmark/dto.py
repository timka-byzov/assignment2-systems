from dataclasses import dataclass
from enum import StrEnum, auto


class BenchmarkType(StrEnum):
    FORWARD = auto()
    FORWARD_BACKWARD = auto()
    FORWARD_BACKWARD_OPTIMIZER = auto()


class ProfilerType(StrEnum):
    TIMING = auto()
    NVTX = auto()


@dataclass
class BenchmarkConfig:
    benchmark_type: BenchmarkType
    warmup_steps: int
    steps: int
    profiler_type: ProfilerType
    workload: str
    batch_size: int = 10
    context_length: int = 512
    dp: int | None = None
    device: str = "cuda"


@dataclass
class BenchmarkResult:
    forward: float = 0.0
    backward: float = 0.0
    optimizer: float = 0.0


@dataclass
class BenchmarkStatistic:
    mean: BenchmarkResult
    std: BenchmarkResult
