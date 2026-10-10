import argparse
import os

import torch
import torch.distributed as dist

from cs336_systems.benchmark.models import BenchmarkConfig, BenchmarkType, ModelConfig, TransfomerConfig
from cs336_systems.benchmark.runner.setup import setup_benchmark
from cs336_systems.benchmark.stat import prettify_results, process_results


def setup():
    assert torch.cuda.is_available()

    if "RANK" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def main():

    parser = argparse.ArgumentParser(prog="benchmark", epilog="Thank you for using MyCLI!")

    parser.add_argument("d_model", type=int)
    parser.add_argument("d_ff", type=int)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("num_heads", type=int)
    parser.add_argument("type")

    parser.add_argument("steps", type=int)
    parser.add_argument("warmup_steps", type=int)

    parser.add_argument("dp", type=int, default=1)

    parser.add_argument("-f", "--forward", action="store_true", help="forward only")

    # parser.add_argument(
    #     "-v", "--verbose", action="store_true", help="Enable verbose logging."
    # )

    args = parser.parse_args()
    tranformer_config = TransfomerConfig(
        vocab_size=10_000,
        context_length=512,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    model_config = ModelConfig(tranformer_config, args.dp)

    config = BenchmarkConfig(
        model_config=model_config,
        benchmark_type=BenchmarkType(args.type),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
    )

    _, _, worls_size = setup()
    mesh = None
    if worls_size > 1:
        mesh = dist.device_mesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(model_config.dp,),
            mesh_dim_names=("dp",),
        )

    model, optimizer, batch = setup_benchmark(config, mesh)

    if args.type == BenchmarkType.NSYS:
        from cs336_systems.benchmark.runner.impl.nvtx import run_bench

        run_bench(config, model, optimizer, batch)

    else:
        from cs336_systems.benchmark.runner.impl.timing import run_bench

        results = run_bench(config, model, optimizer, batch)
        if results and ((mesh and mesh.get_rank() == 0) or not mesh):
            print(prettify_results(process_results(results)))

    print("\nDONE\n\n")

    if mesh:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

# uv run torchrun --standalone --nnodes=1 --nproc-per-node=2 --module cs336_systems.benchmark.main 512 2048 4 8 forward_backward_optimizer 20 5 2
