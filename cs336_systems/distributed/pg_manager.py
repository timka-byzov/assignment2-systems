import os

import torch
import torch.distributed as dist


class ProcessGroupManager:
    def __init__(self, dp_size, tp_size=1, cp_size=1, pp_size=1):
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.global_rank % self.world_size))

        assert tp_size == cp_size == pp_size == 1, "DP is only supported now"
        assert self.world_size == dp_size * tp_size * cp_size * pp_size

        self.grid = torch.arange(self.world_size).view(dp_size, pp_size, cp_size, tp_size)

        self.dp_rank, self.pp_rank, self.cp_rank, self.tp_rank = (self.grid == self.global_rank).nonzero().flatten().tolist()

        
        self.dp_group = dist.new_subgroups_by_enumeration([self.grid[:, p, c, t].tolist() for p in range(pp_size) for c in range(cp_size) for t in range(tp_size)])[0]
