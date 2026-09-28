from dataclasses import asdict

import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

from cs336_systems.benchmark.data import generate_random_batch
from cs336_systems.benchmark.models import BenchmarkConfig, ModelConfig


def init_model(config: ModelConfig) -> BasicsTransformerLM:
    return BasicsTransformerLM(**asdict(config))


def setup_benchmark(config: BenchmarkConfig):
    assert torch.cuda.is_available()
    model = init_model(config.model_config).cuda()
    optimizer = AdamW(model.parameters())
    batch = generate_random_batch(
        config.batch_size,
        config.model_config.vocab_size,
        max_seq_len=config.model_config.context_length,
    ).cuda()

    return model, optimizer, batch
