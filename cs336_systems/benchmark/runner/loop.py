from contextlib import nullcontext

import torch
from torch.cuda import nvtx
from tqdm import tqdm

from cs336_systems.benchmark.context import Context
from cs336_systems.benchmark.dto import BenchmarkConfig, BenchmarkResult, ProfilerType
from cs336_systems.benchmark.runner.impl.nvtx import NvtxStep
from cs336_systems.benchmark.runner.impl.timing import TimingStep
from cs336_systems.benchmark.workload import Workload


class Benchmark:
    def __init__(self, config: BenchmarkConfig, workload: Workload, ctx: Context):
        self.config = config
        self.workload = workload
        self.ctx = ctx
        self.step = TimingStep(ctx.device) if config.profiler_type == ProfilerType.TIMING else NvtxStep()

    def run_once(self):
        return self.step(self.workload, self.config.benchmark_type)

    def warmup(self):
        for _ in tqdm(range(self.config.warmup_steps), desc="WARMUP", disable=self.ctx.rank != 0):
            self.run_once()

    def run(self) -> list[BenchmarkResult] | None:
        self.warmup()
        results = []
        if torch.device(self.ctx.device).type == "cuda":
            torch.cuda.synchronize(self.ctx.device)

        capture = nvtx.range("measure") if self.config.profiler_type == ProfilerType.NVTX else nullcontext()
        with capture:
            for _ in tqdm(range(self.config.steps), desc="BENCHMARK", disable=self.ctx.rank != 0):
                result = self.run_once()
                if result is not None:
                    results.append(result)
            if torch.device(self.ctx.device).type == "cuda":
                torch.cuda.synchronize(self.ctx.device)

        return results if self.config.profiler_type == ProfilerType.TIMING else None
