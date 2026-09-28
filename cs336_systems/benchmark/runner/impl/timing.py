import timeit
from functools import partial

import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from tqdm import tqdm

from cs336_systems.benchmark.models import BenchmarkConfig, BenchmarkResult, BenchmarkType
from cs336_systems.benchmark.runner.common import warmup
from cs336_systems.benchmark.runner.setup import setup_benchmark


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


def run_bench_once(model: BasicsTransformerLM, batch: torch.Tensor, optimizer: torch.optim.Optimizer, benchmark_type: BenchmarkType) -> BenchmarkResult:
    result = BenchmarkResult()

    with TimeManager() as t:
        out = model.forward(batch)
    result.forward = t.elapsed

    if benchmark_type != BenchmarkType.FORWARD:
        loss = cross_entropy(out[:, :-1, ...], batch[:, 1:, ...])
        if optimizer:
            optimizer.zero_grad()

        with TimeManager() as t:
            loss.backward()
        result.backward = t.elapsed

        if benchmark_type == BenchmarkType.FORWARD_BACKWARD_OPTIMIZER:
            with TimeManager() as t:
                optimizer.step()
            result.optimizer = t.elapsed

    return result


def run_bench(config: BenchmarkConfig) -> list[BenchmarkResult] | None:
    model, optimizer, batch = setup_benchmark(config)
    results = []
    params = (model, batch, optimizer, config.benchmark_type)

    warmup(config.warmup_steps, partial(run_bench_once, *params))

    for step in tqdm(range(config.steps)):
        results.append(run_bench_once(*params))

    return results
