from functools import partial

import torch
from cs336_basics.nn_utils import cross_entropy
from torch.cuda import nvtx
from tqdm import tqdm

from cs336_systems.benchmark.models import BenchmarkConfig
from cs336_systems.benchmark.runner.common import warmup


def run_bench_once(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: torch.Tensor):
    with nvtx.range("forward"):
        out = model.forward(batch)

    with nvtx.range("zero_grad_and_loss"):
        optimizer.zero_grad()
        loss = cross_entropy(out[:, :-1, ...], batch[:, 1:, ...])

    with nvtx.range("backward"):
        loss.backward()

    with nvtx.range("optimizer_step"):
        optimizer.step()


def run_bench(config: BenchmarkConfig, model, optimizer, batch) -> None:

    params = (model, optimizer, batch)

    warmup(config.warmup_steps, partial(run_bench_once, *params))

    torch.cuda.synchronize()  # no noise from warmup
    with nvtx.range("measure"):
        for step in tqdm(range(config.steps)):
            run_bench_once(*params)
        torch.cuda.synchronize()  # wait for all kernels to be included in measure
