from cs336_systems.benchmark.context import Context
from cs336_systems.benchmark.workloads.transformer import ModelConfig, TransformerWorkload
from cs336_systems.distributed.ddp import DataParallelNaive


class DataParallelWorkload(TransformerWorkload):
    def __init__(self, config: ModelConfig, ctx: Context) -> None:
        assert ctx.mesh is not None, "DataParallelWorkload requires a DP device mesh"

        super().__init__(config, ctx)
        pg = ctx.mesh.get_group("dp")
        if pg.size() > 1:
            self.model = DataParallelNaive(self.model, pg)
