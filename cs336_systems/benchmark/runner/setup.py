from dataclasses import asdict

import torch
import torch.distributed as dist
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

from cs336_systems.benchmark.data import generate_random_batch
from cs336_systems.benchmark.models import BenchmarkConfig, TransfomerConfig
from cs336_systems.distributed.ddp import DataParallelNaive


def init_model(config: TransfomerConfig, mesh: dist.device_mesh.DeviceMesh | None) -> torch.nn.Module:
    model = BasicsTransformerLM(**asdict(config)).cuda()

    if mesh and mesh.get_group("dp").size() > 1:
        model = DataParallelNaive(
            model,
            mesh.get_group("dp"),
        )

    return model


def setup_benchmark(config: BenchmarkConfig, mesh: dist.device_mesh.DeviceMesh | None):
    transformer_config = config.model_config.tranfsormer_config
    model = init_model(transformer_config, mesh)
    optimizer = AdamW(model.parameters())
    batch = generate_random_batch(
        config.batch_size,
        transformer_config.vocab_size,
        max_seq_len=transformer_config.context_length,
    )

    return model, optimizer, batch
