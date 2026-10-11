import contextlib
from functools import partial

import torch
import torch.distributed as dist

from cs336_systems.distributed.bucket import BucketManager


class _DataParallelBase(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, pg: dist.ProcessGroup, device: str | torch.device):
        super().__init__()

        self.model = model
        self.pg = pg
        self.device = device
        self.require_backward_grad_sync = True  # whether to synchronize gradients during backward pass (GAS moment)

        self._sync_parameters(self.model)

    def _sync_parameters(self, model: torch.nn.Module):
        flat_parameters_buffer = torch.nn.utils.parameters_to_vector(model.parameters())
        dist.broadcast(
            flat_parameters_buffer,
            src=0,
            group=self.pg,
        )
        torch.nn.utils.vector_to_parameters(flat_parameters_buffer, model.parameters())

    @contextlib.contextmanager
    def no_sync(self):
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True


# спизжено из picotron
class DataParallelNaive(_DataParallelBase):
    def __init__(self, model: torch.nn.Module, pg: dist.ProcessGroup, device: str | torch.device):
        super().__init__(model, pg, device)

        self.register_backward_hook(self._allreduce_grads)  # type: ignore

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def register_backward_hook(self, hook):
        for p in self.model.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)

    def _allreduce_grads(self, grad: torch.Tensor):
        # > The hook should not modify its argument, but it can optionally return a new gradient which will be used in place
        out_grad = grad.clone().contiguous()

        if self.require_backward_grad_sync:
            dist.all_reduce(out_grad, op=dist.ReduceOp.SUM, group=self.pg)
            out_grad /= dist.get_world_size(group=self.pg)
        return out_grad


class DataParallelBucket(_DataParallelBase):
    def __init__(self, model: torch.nn.Module, pg: dist.ProcessGroup, device: str | torch.device):
        super().__init__(model, pg, device)
        self.bucket_manager = BucketManager(
            module=self.model,
            device=self.device,
            pg=self.pg,
        )
        self.register_backward_hooks(self._pre_accum_hook, self._post_accum_hook)

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def register_backward_hooks(self, pre_accum_hook, post_accum_hook):
        for p in self.model.parameters():
            if p.requires_grad is True:
                p.register_hook(partial(pre_accum_hook, p))  # джокушка замыкания
                p.register_post_accumulate_grad_hook(post_accum_hook)

    def _pre_accum_hook(self, param: torch.nn.Parameter, grad: torch.Tensor):
        self.bucket_manager.ensure_grad_buffer(param)
        return grad

    def _post_accum_hook(self, param: torch.nn.Parameter):
        if self.require_backward_grad_sync:
            self.bucket_manager.mark_param_as_ready(param)
