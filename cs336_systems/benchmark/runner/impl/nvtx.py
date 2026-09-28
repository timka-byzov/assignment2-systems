from functools import partial

import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from tqdm import tqdm

from cs336_systems.benchmark.models import BenchmarkConfig
from cs336_systems.benchmark.runner.common import warmup
from cs336_systems.benchmark.runner.setup import setup_benchmark


def run_bench_once(model: BasicsTransformerLM, batch: torch.Tensor, optimizer: torch.optim.Optimizer):
    with nvtx.range("forward"):
        out = model.forward(batch)

    with nvtx.range("zero_grad_and_loss"):
        optimizer.zero_grad()
        loss = cross_entropy(out[:, :-1, ...], batch[:, 1:, ...])

    with nvtx.range("backward"):
        loss.backward()

    with nvtx.range("optimizer_step"):
        optimizer.step()


def run_bench(config: BenchmarkConfig) -> None:
    model, optimizer, batch = setup_benchmark(config)

    params = (model, batch, optimizer)

    warmup(config.warmup_steps, partial(run_bench_once, *params))

    torch.cuda.synchronize()  # no noise from warmup
    with nvtx.range("measure"):
        for step in tqdm(range(config.steps)):
            run_bench_once(*params)
        torch.cuda.synchronize()  # wait for all kernels to be included in measure
