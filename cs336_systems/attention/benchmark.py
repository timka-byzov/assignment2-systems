import timeit
from dataclasses import dataclass
from itertools import product

import torch
from jaxtyping import Float
from tqdm import tqdm

from cs336_systems.attention.impl.pytorch import SDPA

BATCH_SIZE = 8
HEAD_SIZES = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
FORWARD_PASSES = 100
BACKWARD_PASSES = 100
WARMUP_STEPS = 5


@dataclass
class AttnParams:
    Q: Float[torch.Tensor, " ... queries d_h"]
    K: Float[torch.Tensor, " ... keys d_h"]
    V: Float[torch.Tensor, " ... keys d_h"]
    is_causal: bool = True


@dataclass
class AttnInputSpec:
    bs: int
    seq_len: int
    d_h: int
    device: str


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


def mean(data: list[int]):
    return sum(data) / len(data)


def make_projection(input_spec: AttnInputSpec, requires_grad: bool = False) -> Float[torch.Tensor, " ... queries d_h"]:
    return torch.rand(
        (
            input_spec.bs,
            input_spec.seq_len,
            input_spec.d_h,
        ),
        requires_grad=requires_grad,
        device=input_spec.device,
    )


def get_attn_params(input_spec: AttnInputSpec, is_causal: bool) -> AttnParams:
    torch.random.manual_seed(0)
    return AttnParams(
        Q=make_projection(input_spec, requires_grad=True),
        K=make_projection(input_spec, requires_grad=True),
        V=make_projection(input_spec, requires_grad=True),
        is_causal=is_causal,
    )


def clear_grads(params: AttnParams):
    params.Q.grad = None
    params.K.grad = None
    params.V.grad = None


def warmup(steps: int, module, params: AttnParams, dO: Float[torch.Tensor, " ... queries d_h"] | None = None):
    for step in range(steps):
        out: torch.Tensor = module.forward(params.Q, params.K, params.V, params.is_causal)
        if dO is not None:
            out.backward(dO)
            clear_grads(params)
    return out


@dataclass
class BenchmarkResult:
    spec: AttnInputSpec
    forward_mem: int
    forward_time: float
    backward_time: float


def run_bench_once(spec: AttnInputSpec, module: torch.nn.Module) -> BenchmarkResult:
    params = get_attn_params(spec, is_causal=False)

    warmup(WARMUP_STEPS, module, params)
    fwd_timings = []
    pre_fwd_allocated = torch.cuda.memory_allocated()
    out = None
    for step in range(FORWARD_PASSES):
        if out is not None:
            del out  # release buffer and graph data to prevent OOM
        with TimeManager() as t:
            out = module.forward(params.Q, params.K, params.V, params.is_causal)
        fwd_timings.append(t.elapsed)
    fwd_allocated = torch.cuda.memory_allocated() - pre_fwd_allocated  # out graph is alive at the moment of measure
    del out  # release graph data

    dO = make_projection(spec)
    warmup(WARMUP_STEPS, module, params, dO)
    bwd_timings = []
    out = None
    for step in range(BACKWARD_PASSES):
        if out is not None:
            del out  # release buffer to prevent OOM
        out = module.forward(params.Q, params.K, params.V, params.is_causal)
        with TimeManager() as t:
            out.backward(dO)  # releases graph data auto
        bwd_timings.append(t.elapsed)
        clear_grads(params)
    del out

    return BenchmarkResult(
        spec=spec,
        forward_mem=fwd_allocated,
        forward_time=mean(fwd_timings),
        backward_time=mean(bwd_timings),
    )


# vibecode
def print_table(results: dict[tuple, BenchmarkResult]):
    print(f"{'d_h':>5} {'seq_len':>8} {'forward ms':>11} {'backward ms':>12} {'Δmem MiB':>10}")
    print("-" * 52)

    for (d_h, seq_len), result in sorted(results.items()):
        print(f"{d_h:5d} {seq_len:8d} {result.forward_time * 1000:11.3f} {result.backward_time * 1000:12.3f} {result.forward_mem / 2**20:10.1f}")


def run_bench(module: torch.nn.Module, device: str):
    result_dict = {}

    for d_h, seq_len in tqdm(
        product(HEAD_SIZES, SEQ_LENS),
        total=len(HEAD_SIZES) * len(SEQ_LENS),
        desc="benchmarking",
    ):
        spec = AttnInputSpec(bs=BATCH_SIZE, seq_len=seq_len, d_h=d_h, device=device)
        try:
            res = run_bench_once(spec, module)
            result_dict[(d_h, seq_len)] = res
        except torch.cuda.OutOfMemoryError:
            print(f"OOM on {d_h} {seq_len}")

    print_table(result_dict)


def main():
    assert torch.cuda.is_available()
    device = "cuda:0"

    run_bench(SDPA(), device=device)


main()
