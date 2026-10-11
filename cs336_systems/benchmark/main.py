import argparse

from cs336_systems.benchmark.dto import BenchmarkConfig, BenchmarkType, ProfilerType
from cs336_systems.benchmark.runtime import Runtime
from cs336_systems.benchmark.workloads.presets import _TRANSFORMERS


def main():
    parser = argparse.ArgumentParser(prog="benchmark")
    parser.add_argument("workload", choices=_TRANSFORMERS)
    parser.add_argument("type", choices=list(BenchmarkType))
    parser.add_argument("steps", type=int)
    parser.add_argument("warmup_steps", type=int)
    parser.add_argument("-dp", "--dp", type=int, help="defaults to torchrun WORLD_SIZE")
    parser.add_argument("--profiler", dest="profiler_type", choices=list(ProfilerType), default=ProfilerType.TIMING)
    parser.add_argument("--batch-size", type=int, default=10, help="local batch size per rank")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("-f", "--forward", action="store_true", help="forward only")
    args = parser.parse_args()

    config = BenchmarkConfig(
        benchmark_type=BenchmarkType.FORWARD if args.forward else BenchmarkType(args.type),
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        profiler_type=ProfilerType(args.profiler_type),
        workload=args.workload,
        batch_size=args.batch_size,
        context_length=args.context_length,
        dp=args.dp,
        device=args.device,
    )
    Runtime.run(config)


if __name__ == "__main__":
    main()
