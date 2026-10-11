from torch.cuda import nvtx

from cs336_systems.benchmark.dto import BenchmarkType
from cs336_systems.benchmark.runner.step import ProfilerStep
from cs336_systems.benchmark.workload import Workload


class NvtxStep(ProfilerStep):
    def __call__(self, workload: Workload, benchmark_type: BenchmarkType) -> None:
        with nvtx.range("forward"):
            workload.forward()

        if benchmark_type != BenchmarkType.FORWARD:
            with nvtx.range("backward"):
                workload.backward()

            if benchmark_type == BenchmarkType.FORWARD_BACKWARD_OPTIMIZER:
                with nvtx.range("optimizer_step"):
                    workload.optimizer_step()
