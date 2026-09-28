import argparse


from cs336_systems.benchmark.models import ModelConfig, BenchmarkType, BenchmarkConfig
from cs336_systems.benchmark.stat import prettify_results, process_results
from cs336_systems.benchmark.runner import run_bench


def main():
    parser = argparse.ArgumentParser(prog="benchmark", epilog="Thank you for using MyCLI!")

    parser.add_argument("d_model", type=int)
    parser.add_argument("d_ff", type=int)
    parser.add_argument("num_layers", type=int)
    parser.add_argument("num_heads", type=int)
    parser.add_argument("type")

    parser.add_argument("steps", type=int)
    parser.add_argument("warmup_steps", type=int)

    parser.add_argument("-f", "--forward", action="store_true", help="forward only")

    # parser.add_argument(
    #     "-v", "--verbose", action="store_true", help="Enable verbose logging."
    # )

    args = parser.parse_args()

    results = run_bench(
        BenchmarkConfig(
            model_config=ModelConfig(
                vocab_size=10_000,
                context_length=512,
                d_model=args.d_model,
                d_ff=args.d_ff,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
            ),
            benchmark_type=BenchmarkType(args.type.lower()),
            steps=args.steps,
            warmup_steps=args.warmup_steps,
        )
    )
    if results:
        print(prettify_results(process_results(results)))
    print("\n\nDONE")


if __name__ == "__main__":
    main()
