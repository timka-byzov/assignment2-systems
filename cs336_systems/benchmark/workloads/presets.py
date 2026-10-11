from dataclasses import replace

from cs336_systems.benchmark.context import Context
from cs336_systems.benchmark.dto import DDPType
from cs336_systems.benchmark.workloads.dp import DataParallelWorkload
from cs336_systems.benchmark.workloads.transformer import ModelConfig, TransfomerConfig, TransformerWorkload

SMALL = ModelConfig(TransfomerConfig(vocab_size=10_000, context_length=512, d_model=768, num_layers=12, num_heads=12, d_ff=3072), batch_size=10)
MEDIUM = ModelConfig(TransfomerConfig(vocab_size=10_000, context_length=512, d_model=1024, num_layers=24, num_heads=16, d_ff=4096), batch_size=10)
LARGE = ModelConfig(TransfomerConfig(vocab_size=10_000, context_length=512, d_model=1280, num_layers=36, num_heads=20, d_ff=5120), batch_size=10)
XL = ModelConfig(TransfomerConfig(vocab_size=10_000, context_length=512, d_model=2560, num_layers=32, num_heads=32, d_ff=10240), batch_size=10)
TEN_B = ModelConfig(TransfomerConfig(vocab_size=10_000, context_length=512, d_model=4608, num_layers=50, num_heads=36, d_ff=12288), batch_size=10)

_TRANSFORMERS = {"small": SMALL, "medium": MEDIUM, "large": LARGE, "xl": XL, "10b": TEN_B}


def get_workload(name: str, ctx: Context, batch_size: int = 10, context_length: int = 512, ddp_type: DDPType = DDPType.NAIVE) -> TransformerWorkload:
    config = _TRANSFORMERS[name]
    config = replace(config, batch_size=batch_size, tranfsormer_config=replace(config.tranfsormer_config, context_length=context_length))
    if ctx.mesh is not None and ctx.mesh.get_group("dp").size() > 1:
        return DataParallelWorkload(config, ctx, ddp_type=ddp_type)
    return TransformerWorkload(config, ctx)
