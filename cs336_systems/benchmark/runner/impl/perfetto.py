import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, _ExperimentalConfig, profile, record_function

from cs336_systems.benchmark.context import Context
from cs336_systems.benchmark.dto import BenchmarkConfig, BenchmarkType
from cs336_systems.benchmark.runner.step import ProfilerStep
from cs336_systems.benchmark.workload import Workload


class PerfettoStep(ProfilerStep):
    profile_memory = False

    def __init__(self, config: BenchmarkConfig, ctx: Context):
        self.config = config
        self.ctx = ctx

    @contextmanager
    def capture(self):
        if self.ctx.rank != 0:
            yield
            return

        activities = [ProfilerActivity.CPU]
        if torch.device(self.ctx.device).type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            profile_memory=self.profile_memory,
            with_stack=self.profile_memory,
            experimental_config=_ExperimentalConfig(verbose=self.profile_memory),
        ) as profiler:
            with record_function("measure"):
                yield

        trace_dir = Path(self.config.trace_dir).expanduser() if self.config.trace_dir is not None else Path.home() / "wip" / "benchmarks" / "perfetto"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{self.config.workload}_{self.config.profiler_type}_rank0_{time.time_ns()}.json"
        profiler.export_chrome_trace(str(trace_path))
        if self.profile_memory:
            self._add_memory_counters(trace_path)
        print(f"Perfetto trace: {trace_path.resolve()}")

    def __call__(self, workload: Workload, benchmark_type: BenchmarkType) -> None:
        with record_function("step"):
            with record_function("forward"):
                workload.forward()

            if benchmark_type != BenchmarkType.FORWARD:
                with record_function("backward"):
                    workload.backward()

                if benchmark_type == BenchmarkType.FORWARD_BACKWARD_OPTIMIZER:
                    with record_function("optimizer_step"):
                        workload.optimizer_step()

    @staticmethod
    def _add_memory_counters(trace_path: Path):
        trace = json.loads(trace_path.read_text())
        counters = []
        for event in trace["traceEvents"]:
            if event.get("name") != "[memory]":
                continue
            args = event["args"]
            device = "cpu" if args["Device Type"] == 0 else f"cuda:{args['Device Id']}"
            counters.append(
                {
                    "name": f"{device} memory",
                    "cat": "memory",
                    "ph": "C",
                    "pid": event["pid"],
                    "tid": event["tid"],
                    "ts": event["ts"],
                    "args": {"allocated MiB": args["Total Allocated"] / 1024**2, "reserved MiB": args["Total Reserved"] / 1024**2},
                }
            )
        trace["traceEvents"].extend(counters)
        trace_path.write_text(json.dumps(trace))


class PerfettoComputeStep(PerfettoStep):
    pass


class PerfettoMemoryStep(PerfettoStep):
    profile_memory = True
