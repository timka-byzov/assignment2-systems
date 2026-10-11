import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from cs336_systems.benchmark.context import Context
from cs336_systems.benchmark.dto import BenchmarkConfig, ProfilerType
from cs336_systems.benchmark.runner.loop import Benchmark
from cs336_systems.benchmark.stat import prettify_results, process_results
from cs336_systems.benchmark.workloads.presets import get_workload


class Runtime:
    @staticmethod
    def setup(config: BenchmarkConfig) -> Context:
        assert config.steps >= 1 and config.warmup_steps >= 0, "steps must be positive and warmup_steps must be non-negative"
        assert config.batch_size >= 1 and config.context_length >= 2, "batch_size must be positive and context_length must be at least 2"

        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dp = world_size if config.dp is None else config.dp

        device = torch.device(config.device)
        assert device.type in ("cpu", "cuda"), "Only cpu and cuda devices are supported"
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            device = torch.device("cuda", local_rank if "RANK" in os.environ else (device.index or 0))
            torch.cuda.set_device(device)
        assert config.profiler_type != ProfilerType.NVTX or device.type == "cuda", "NVTX profiling requires CUDA"

        mesh = None
        if world_size > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
            mesh = init_device_mesh(device.type, (dp,), mesh_dim_names=("dp",))

        return Context(device=device, rank=rank, world_size=world_size, mesh=mesh)

    @staticmethod
    def run(config: BenchmarkConfig):
        owns_process_group = not dist.is_initialized()
        try:
            ctx = Runtime.setup(config)
            workload = get_workload(config.workload, ctx, config.batch_size, config.context_length, ddp_type=config.ddp_type)
            results = Benchmark(config, workload, ctx).run()
            if ctx.rank == 0:
                if results is not None:
                    print(prettify_results(process_results(results)))
                print("\nDONE\n")
            return results
        finally:
            if owns_process_group and dist.is_initialized():
                dist.destroy_process_group()
