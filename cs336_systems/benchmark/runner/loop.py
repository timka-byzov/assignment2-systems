import torch
from tqdm import tqdm

from cs336_systems.benchmark.context import Context
from cs336_systems.benchmark.dto import BenchmarkConfig, BenchmarkResult, ProfilerType
from cs336_systems.benchmark.runner.impl.nvtx import NvtxStep
from cs336_systems.benchmark.runner.impl.perfetto import PerfettoComputeStep, PerfettoMemoryStep
from cs336_systems.benchmark.runner.impl.timing import TimingStep
from cs336_systems.benchmark.workload import Workload


class Benchmark:
    def __init__(self, config: BenchmarkConfig, workload: Workload, ctx: Context):
        self.config = config
        self.workload = workload
        self.ctx = ctx
        if config.profiler_type == ProfilerType.TIMING:
            self.step = TimingStep(ctx.device)
        elif config.profiler_type == ProfilerType.NVTX:
            self.step = NvtxStep()
        elif config.profiler_type == ProfilerType.PERFETTO_COMPUTE:
            self.step = PerfettoComputeStep(config, ctx)
        else:
            assert config.profiler_type == ProfilerType.PERFETTO_MEMORY, "Unsupported profiler"
            self.step = PerfettoMemoryStep(config, ctx)

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

        with self.step.capture():
            for _ in tqdm(range(self.config.steps), desc="BENCHMARK", disable=self.ctx.rank != 0):
                result = self.run_once()
                if result is not None:
                    results.append(result)
            if torch.device(self.ctx.device).type == "cuda":
                torch.cuda.synchronize(self.ctx.device)

        return results if self.config.profiler_type == ProfilerType.TIMING else None
