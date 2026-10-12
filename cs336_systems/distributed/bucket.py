from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist


@dataclass
class ParamInfo:
    id: int
    start: int
    numel: int
    shape: tuple
    ready: bool = False


@dataclass
class SyncJob:
    bucket: "Bucket"
    handle: dist.Work


class BucketManager:
    def __init__(self, module: torch.nn.Module, device: str | torch.device, pg: dist.ProcessGroup, max_bucket_size_mb: int = 1024, async_op: bool = True):
        self.max_bucket_size_b = max_bucket_size_mb * 1024 * 1024
        self.device = device
        self.pg = pg
        self.bucket_factory = partial(Bucket, pg=pg, device=device, async_op=async_op)
        self.buckets, self.param_id_to_bucket = self._split_params_into_buckets(module.parameters())
        self.sync_jobs: list[SyncJob] = []

    def _split_params_into_buckets(self, params: Iterable[torch.nn.Parameter]) -> tuple[list["Bucket"], dict[int, "Bucket"]]:

        curr_bucket_size_numel = 0
        curr_bucket_size_dtype = None
        curr_bucket_params_info = []
        buckets: list[Bucket] = []
        param_id_to_bucket = {}

        for p in params:
            if p.requires_grad:
                if curr_bucket_size_dtype is None:
                    curr_bucket_size_dtype = p.dtype

                p_size_b = p.numel() * p.dtype.itemsize

                assert p_size_b <= self.max_bucket_size_b

                if curr_bucket_size_dtype != p.dtype or curr_bucket_size_numel * curr_bucket_size_dtype.itemsize + p_size_b > self.max_bucket_size_b:
                    bucket = self.bucket_factory(curr_bucket_params_info, curr_bucket_size_dtype)
                    param_id_to_bucket.update({curr_p.id: bucket for curr_p in curr_bucket_params_info})
                    buckets.append(bucket)

                    curr_bucket_size_numel = 0
                    curr_bucket_params_info = []

                curr_bucket_size_dtype = p.dtype
                curr_bucket_params_info.append(
                    ParamInfo(
                        start=curr_bucket_size_numel,
                        numel=p.numel(),
                        shape=p.shape,
                        id=id(p),
                    )
                )
                curr_bucket_size_numel += p.numel()

        if curr_bucket_size_numel > 0:
            bucket = self.bucket_factory(curr_bucket_params_info, curr_bucket_size_dtype)
            param_id_to_bucket.update({curr_p.id: bucket for curr_p in curr_bucket_params_info})
            buckets.append(bucket)

        return buckets, param_id_to_bucket

    def ensure_grad_buffer(self, parameter: torch.nn.Parameter):
        if parameter.grad is None:
            p_id = id(parameter)
            bucket = self.param_id_to_bucket[p_id]
            parameter.grad = bucket.prepare_grad_view(p_id)

    def mark_param_as_ready(self, parameter: torch.nn.Parameter):
        p_id = id(parameter)
        bucket = self.param_id_to_bucket[p_id]
        if (job := bucket.mark_param_as_ready(p_id)) is not None:
            self.sync_jobs.append(job)

    def finish_grad_sync(self):
        for job in self.sync_jobs:
            job.handle.wait()
            job.bucket.finish_sync_buffer()

        self.sync_jobs.clear()


class Bucket:
    def __init__(
        self,
        params_info: list[ParamInfo],
        dtype: torch.dtype,
        device: str | torch.device,
        pg: dist.ProcessGroup,
        async_op: bool,
    ):
        self.params_info = params_info
        self.device = device
        self.dtype = dtype
        self.pg = pg
        self.async_op = async_op

        self.numel = sum(p.numel for p in params_info)  # только фактический буффер коммуницирует (надо проверять профит)
        self.buffer = torch.zeros((self.numel,), device=self.device, dtype=self.dtype)
        self.param_id_to_info = {p.id: p for p in params_info}

    def _mark_bucket_not_ready(self):
        for p in self.params_info:
            p.ready = False

    def is_bucket_ready(self):
        return all(p.ready is True for p in self.params_info)

    def prepare_grad_view(self, param_id: int) -> torch.Tensor:
        param_info = self.param_id_to_info[param_id]
        return self.buffer[param_info.start : param_info.start + param_info.numel].view(param_info.shape).zero_()

    def mark_param_as_ready(self, param_id) -> SyncJob | None:
        self.param_id_to_info[param_id].ready = True
        if self.is_bucket_ready():
            if (handle := dist.all_reduce(self.buffer, group=self.pg, async_op=self.async_op)) is not None:
                return SyncJob(self, handle)
            else:
                self.finish_sync_buffer()

    def finish_sync_buffer(self):
        self.buffer /= self.pg.size()
        self._mark_bucket_not_ready()
