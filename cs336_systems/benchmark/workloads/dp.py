from cs336_systems.benchmark.context import Context
from cs336_systems.benchmark.dto import DDPType
from cs336_systems.benchmark.workloads.transformer import ModelConfig, TransformerWorkload
from cs336_systems.distributed.ddp import (
    DataParallelBucket,
    DataParallelNaiveImmediateSync,
    DataParallelNaiveLateSync,
    DataParallelPro,
)


class DataParallelWorkload(TransformerWorkload):
    def __init__(self, config: ModelConfig, ctx: Context, ddp_type: DDPType = DDPType.NAIVE) -> None:
        assert ctx.mesh is not None, "DataParallelWorkload requires a DP device mesh"

        super().__init__(config, ctx)
        pg = ctx.mesh.get_group("dp")
        if pg.size() > 1:
            ddp_class = {
                DDPType.NAIVE: DataParallelNaiveImmediateSync,
                DDPType.NAIVE_LATE: DataParallelNaiveLateSync,
                DDPType.BUCKET: DataParallelBucket,
                DDPType.PRO: DataParallelPro,
            }[ddp_type]
            self.model = ddp_class(self.model, pg, device=ctx.device)

    def backward(self, gas=False):
        if not gas:
            self.optimizer.zero_grad()
        assert self.loss is not None, "call forward"
        self.model.backward(self.loss)
        self.loss = None
