from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import DeviceMesh


@dataclass
class Context:
    device: str | torch.device
    rank: int
    world_size: int
    mesh: DeviceMesh | None = None
