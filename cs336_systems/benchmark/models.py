from dataclasses import dataclass
from enum import auto, StrEnum


class BenchmarkType(StrEnum):
    FORWARD = auto()
    FORWARD_BACKWARD = auto()
    FORWARD_BACKWARD_OPTIMIZER = auto()


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float | None = 10_000.0


@dataclass
class BenchmarkConfig:
    model_config: ModelConfig
    benchmark_type: BenchmarkType
    batch_size: int = 10
    warmup_steps: int = 5
    steps: int = 1000


@dataclass
class BenchmarkResult:
    forward: float = 0.0
    backward: float = 0.0
    optimizer: float = 0.0


@dataclass
class BenchmarkStatistic:
    mean: BenchmarkResult
    std: BenchmarkResult
