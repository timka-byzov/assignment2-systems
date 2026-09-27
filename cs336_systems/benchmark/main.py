from cs336_basics.model import BasicsTransformerLM
from dataclasses import dataclass, asdict
import argparse

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import torch
from cs336_systems.benchmark.data import generate_random_batch
from enum import StrEnum, auto
import timeit


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
    steps: int = 10


@dataclass
class BenchmarkResult:
    forward: float = 0.0
    backward: float = 0.0
    optimizer: float = 0.0


def init_model(config: ModelConfig) -> BasicsTransformerLM:
    return BasicsTransformerLM(**asdict(config))


class TimeManager:
    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self.star_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        self.elapsed = timeit.default_timer() - self.star_time


def run_bench_once(model: BasicsTransformerLM, batch: torch.Tensor, bencmark_type: BenchmarkType, optimizer: torch.optim.Optimizer) -> BenchmarkResult:
    result = BenchmarkResult()

    with TimeManager() as t:
        out = model.forward(batch)
    result.forward = t.elapsed

    if bencmark_type != BenchmarkType.FORWARD:
        loss = cross_entropy(out[:, :-1, ...], batch[:, 1:, ...])
        if optimizer:
            optimizer.zero_grad()

        with TimeManager() as t:
            loss.backward()
        result.backward = t.elapsed

        if bencmark_type == BenchmarkType.FORWARD_BACKWARD_OPTIMIZER:
            with TimeManager() as t:
                optimizer.step()
            result.optimizer = t.elapsed

    return result


def _process_results(results: list[BenchmarkResult], steps: int) -> BenchmarkResult:
    result = BenchmarkResult()
    for x in results:
        result.forward += x.forward
        result.backward += x.backward
        result.optimizer += x.optimizer
    result.forward /= steps
    result.backward /= steps
    result.optimizer /= steps

    return result


def run_bench(config: BenchmarkConfig):
    assert torch.cuda.is_available()
    model = init_model(config.model_config).cuda()
    optimizer = AdamW(model.parameters())
    results = []

    batch = generate_random_batch(
        config.batch_size,
        config.model_config.vocab_size,
        max_seq_len=config.model_config.context_length,
    ).cuda()

    for step in range(config.warmup_steps + config.steps):
        result = run_bench_once(model, batch, config.benchmark_type, optimizer)

        if step < config.warmup_steps:
            continue

        results.append(result)

    return _process_results(results, config.steps)


def main():
    parser = argparse.ArgumentParser(prog="benchmark", epilog="Thank you for using MyCLI!")

    parser.add_argument("d_model")
    parser.add_argument("d_ff")
    parser.add_argument("num_layers")
    parser.add_argument("num_heads")
    parser.add_argument("type")

    parser.add_argument("-f", "--forward", action="store_true", help="forward only")

    # parser.add_argument(
    #     "-c", "--count", type=int, default=1, help="Number of repetitions."
    # )

    # parser.add_argument(
    #     "-v", "--verbose", action="store_true", help="Enable verbose logging."
    # )

    args = parser.parse_args()

    print(
        run_bench(
            BenchmarkConfig(
                model_config=ModelConfig(
                    vocab_size=10_000,
                    context_length=512,
                    d_model=int(args.d_model),
                    d_ff=int(args.d_ff),
                    num_layers=int(args.num_layers),
                    num_heads=int(args.num_heads),
                ),
                benchmark_type=BenchmarkType(args.type.lower()),
            )
        )
    )


if __name__ == "__main__":
    main()
