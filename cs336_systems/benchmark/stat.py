import torch

from cs336_systems.benchmark.models import BenchmarkResult, BenchmarkStatistic


def _get_mean_and_std(data: list[float]) -> tuple[float, float]:
    t = torch.tensor(data)
    return float(t.mean()), float(t.std())


def process_results(results: list[BenchmarkResult]) -> BenchmarkStatistic:

    fwd_mean, fwd_std = _get_mean_and_std([r.forward for r in results])
    bwd_mean, bwd_std = _get_mean_and_std([r.backward for r in results])
    opt_mean, opt_std = _get_mean_and_std([r.optimizer for r in results])

    return BenchmarkStatistic(
        mean=BenchmarkResult(forward=fwd_mean, backward=bwd_mean, optimizer=opt_mean), std=BenchmarkResult(forward=fwd_std, backward=bwd_std, optimizer=opt_std)
    )


# vibecode
def prettify_results(stat: BenchmarkStatistic):
    return (
        f"Forward:   {stat.mean.forward * 1000:8.2f} ± {stat.std.forward * 1000:6.2f} ms\n"
        f"Backward:  {stat.mean.backward * 1000:8.2f} ± {stat.std.backward * 1000:6.2f} ms\n"
        f"Optimizer: {stat.mean.optimizer * 1000:8.2f} ± {stat.std.optimizer * 1000:6.2f} ms"
    )
