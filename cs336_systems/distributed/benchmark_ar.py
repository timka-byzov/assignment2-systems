import os
import timeit
from pathlib import Path

import torch
import torch.distributed as dist

FP_32_b = 4
BACKEND = "nccl"

TRANSFER_SIZES_MB = [1, 10, 100, 1000]
WARMUP_STEPS = 5


# vibecode plot
def plot_communication_times(measurements: list[tuple[float, float]] | torch.Tensor, name: str):

    import plotly.graph_objects as go

    output_path = Path(__file__).parent / "logs" / f"{name}.png"

    if isinstance(measurements, torch.Tensor):
        measurements = measurements.detach().cpu().tolist()
    sizes, times = zip(*measurements)
    fig = go.Figure(go.Bar(x=sizes, y=times))
    fig.update_layout(xaxis_type="category", xaxis_title="Message size (MiB)", yaxis_title="Communication time (s)")
    fig.write_image(output_path)
    return fig


class TimeManager:
    def __init__(self, device: str) -> None:
        self.elapsed: float = 0.0
        self.device = device

    def __enter__(self):
        if self.device != "cpu":
            torch.cuda.synchronize(self.device)
        self.star_time = timeit.default_timer()
        return self

    def __exit__(self, *args):
        if self.device != "cpu":
            torch.cuda.synchronize(self.device)
        self.elapsed = timeit.default_timer() - self.star_time


def _create_data(size_mb: int, device: str) -> torch.Tensor:
    num_elements = size_mb * 1024 * 1024 // FP_32_b
    return torch.zeros(num_elements, dtype=torch.float32).to(device)


def communication_round(device, data: torch.Tensor) -> float:

    with TimeManager(device) as t:
        dist.all_reduce(data, async_op=False)  # False is default. False = compute stream wating for comm. stream ends

    return t.elapsed


def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    assert local_rank == rank, "Single node training!"

    device = f"cuda:{local_rank}" if BACKEND == "nccl" else "cpu"
    dist.init_process_group(BACKEND, rank=rank, world_size=world_size)
    return rank, world_size, device


def warmup(rank, device, buffer):
    for warmup_step in range(WARMUP_STEPS):
        if rank == 0:
            print(f"Warmup step {warmup_step}")
        for ts_mb in TRANSFER_SIZES_MB:
            num_elements = ts_mb * 1024 * 1024 // FP_32_b
            communication_round(device, data=buffer[:num_elements])


def benchmark(rank, world_size, device, buffer):
    measurements = []

    for ts_mb in TRANSFER_SIZES_MB:
        num_elements = ts_mb * 1024 * 1024 // FP_32_b
        elapsed = torch.tensor(communication_round(device, data=buffer[:num_elements])).to(device)
        overall_elapsed = None
        if rank == 0:
            overall_elapsed = [torch.zeros_like(elapsed, device=device) for _ in range(world_size)]
        dist.gather(elapsed, overall_elapsed)

        if rank == 0:
            median = torch.tensor(overall_elapsed).mean()
            measurements.append((ts_mb, median.detach().cpu()))
    return measurements


def run():
    rank, world_size, device = setup()
    buffer = _create_data(1000, device)
    warmup(rank, device, buffer)
    measurements = benchmark(rank, world_size, device, buffer)

    dist.destroy_process_group()

    if rank == 0:
        plot_communication_times(measurements, name=f"{BACKEND}_{world_size}")


if __name__ == "__main__":
    run()

# torchrun --standalone --nnodes=1 --nproc-per-node=4 cs336_systems/distributed/benchmark_ar.py
