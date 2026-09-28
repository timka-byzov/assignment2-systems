from cs336_basics.model import BasicsTransformerLM
from dataclasses import asdict
import argparse
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import torch
from cs336_systems.benchmark.data import generate_random_batch
import timeit
from cs336_systems.benchmark.models import ModelConfig, BenchmarkType, BenchmarkResult, BenchmarkConfig
from cs336_systems.benchmark.stat import prettify_results, process_results
from tqdm import tqdm


def init_model(config: ModelConfig) -> BasicsTransformerLM:
    return BasicsTransformerLM(**asdict(config))


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


def run_bench_once(model: BasicsTransformerLM, batch: torch.Tensor, bencmark_type: BenchmarkType, optimizer: torch.optim.Optimizer) -> BenchmarkResult:
    result = BenchmarkResult()

    with TimeManager() as t:
        out = model.forward(batch)
    result.forward = t.elapsed

    if bencmark_type != BenchmarkType.FORWARD:
        loss = cross_entropy(out[:, :-1, ...], batch[:, 1:, ...])
        if optimizer:
            optimizer.zero_grad()

        with TimeManager() as t:
            loss.backward()
        result.backward = t.elapsed

        if bencmark_type == BenchmarkType.FORWARD_BACKWARD_OPTIMIZER:
            with TimeManager() as t:
                optimizer.step()
            result.optimizer = t.elapsed

    return result


def run_bench(config: BenchmarkConfig) -> list[BenchmarkResult]:
    assert torch.cuda.is_available()
    model = init_model(config.model_config).cuda()
    optimizer = AdamW(model.parameters())
    results = []

    batch = generate_random_batch(
        config.batch_size,
        config.model_config.vocab_size,
        max_seq_len=config.model_config.context_length,
    ).cuda()

    for step in tqdm(range(config.warmup_steps + config.steps)):
        result = run_bench_once(model, batch, config.benchmark_type, optimizer)

        if step < config.warmup_steps:
            continue

        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(prog="benchmark", epilog="Thank you for using MyCLI!")

    parser.add_argument("d_model")
    parser.add_argument("d_ff")
    parser.add_argument("num_layers")
    parser.add_argument("num_heads")
    parser.add_argument("type")

    parser.add_argument("-f", "--forward", action="store_true", help="forward only")

    # parser.add_argument(
    #     "-c", "--count", type=int, default=1, help="Number of repetitions."
    # )

    # parser.add_argument(
    #     "-v", "--verbose", action="store_true", help="Enable verbose logging."
    # )

    args = parser.parse_args()

    results = run_bench(
        BenchmarkConfig(
            model_config=ModelConfig(
                vocab_size=10_000,
                context_length=512,
                d_model=int(args.d_model),
                d_ff=int(args.d_ff),
                num_layers=int(args.num_layers),
                num_heads=int(args.num_heads),
            ),
            benchmark_type=BenchmarkType(args.type.lower()),
        )
    )

    print(prettify_results(process_results(results)))


if __name__ == "__main__":
    main()
