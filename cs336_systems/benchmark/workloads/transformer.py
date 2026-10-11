from dataclasses import asdict, dataclass

import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_systems.benchmark.context import Context
from cs336_systems.benchmark.data import generate_random_batch
from cs336_systems.benchmark.workload import Workload


@dataclass
class TransfomerConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float | None = 10_000.0


@dataclass
class ModelConfig:
    tranfsormer_config: TransfomerConfig
    batch_size: int


def init_model(config: TransfomerConfig, ctx: Context) -> torch.nn.Module:
    return BasicsTransformerLM(**asdict(config)).to(ctx.device)


class TransformerWorkload(Workload):
    def __init__(self, config: ModelConfig, ctx: Context) -> None:
        self.model = init_model(config.tranfsormer_config, ctx)
        self.optimizer = AdamW(self.model.parameters())
        self.loss = None
        self.batch = generate_random_batch(
            config.batch_size,
            config.tranfsormer_config.vocab_size,
            max_seq_len=config.tranfsormer_config.context_length,
            device=ctx.device,
        )

    def forward(self):
        out = self.model(self.batch)
        self.loss = cross_entropy(out[:, :-1, ...], self.batch[:, 1:, ...])

    def backward(self, gas=False):
        if not gas:
            self.optimizer.zero_grad()
        assert self.loss is not None, "call forward"
        self.loss.backward()
        self.loss = None

    def optimizer_step(self):
        self.optimizer.step()
