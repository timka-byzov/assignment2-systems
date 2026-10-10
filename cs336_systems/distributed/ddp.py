import contextlib

import torch
import torch.distributed as dist


# спизжено из picotron
class DataParallelNaive(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, pg: dist.ProcessGroup):
        super().__init__()

        self.model = model
        self.pg = pg
        self.require_backward_grad_sync = True  # whether to synchronize gradients during backward pass (GAS moment)

        self._sync_parameters(self.model)
        self.register_backward_hook(self._allreduce_grads)  # type: ignore

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def broadcast_model(self):
        if self.rank == 0:
            pass

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

    @contextlib.contextmanager
    def no_sync(self):
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True

    def _sync_parameters(self, model: torch.nn.Module):
        flat_parameters_buffer = torch.nn.utils.parameters_to_vector(model.parameters())
        dist.broadcast(
            flat_parameters_buffer,
            src=0,
            group=self.pg,
        )
        torch.nn.utils.vector_to_parameters(flat_parameters_buffer, model.parameters())
