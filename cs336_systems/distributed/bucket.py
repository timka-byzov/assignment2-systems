from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ParamInfo:
    id: int
    start: int
    numel: int
    shape: tuple
    ready: bool = False


class BucketManager:
    # лениво разбить и инициализировать буффер только когда входим в бакет. при обновлении градиента по бакету - удалять буфер бакета. единый буфер под все бакеты не нужен, ведь существует gas.
    def __init__(self, module: torch.nn.Module, device: str | torch.device, pg: dist.ProcessGroup, max_bucket_size_mb: int = 1024):
        self.max_bucket_size_b = max_bucket_size_mb * 1024 * 1024
        self.device = device
        self.pg = pg

        self.buckets, self.param_id_to_bucket = self._split_params_into_buckets(module.parameters())

    def _split_params_into_buckets(self, params: Iterable[torch.nn.Parameter]) -> tuple[list["Bucket"], dict[int, "Bucket"]]:

        curr_bucket_size_numel = 0
        curr_bucket_size_dtype = None
        curr_bucket_params_info = []
        buckets: list[Bucket] = []
        param_id_to_bucket = dict()

        for p in params:
            if p.requires_grad:
                if curr_bucket_size_dtype is None:
                    curr_bucket_size_dtype = p.dtype

                p_size_b = p.numel() * p.dtype.itemsize

                assert p_size_b <= self.max_bucket_size_b

                if curr_bucket_size_dtype != p.dtype or curr_bucket_size_numel * curr_bucket_size_dtype.itemsize + p_size_b > self.max_bucket_size_b:
                    bucket = Bucket(self.max_bucket_size_b, self.device, curr_bucket_params_info, curr_bucket_size_dtype, self.pg)
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
            bucket = Bucket(self.max_bucket_size_b, self.device, curr_bucket_params_info, curr_bucket_size_dtype, self.pg)
            param_id_to_bucket.update({curr_p.id: bucket for curr_p in curr_bucket_params_info})
            buckets.append(bucket)

        return buckets, param_id_to_bucket

    def ensure_grad_buffer(self, parameter: torch.nn.Parameter):
        if parameter.grad is None:
            p_id = id(parameter)
            bucket = self.param_id_to_bucket[p_id]
            parameter.grad = bucket.get_grad_buffer(p_id)

    def mark_param_as_ready(self, parameter: torch.nn.Parameter):
        p_id = id(parameter)
        bucket = self.param_id_to_bucket[p_id]
        bucket.mark_param_as_ready(p_id)


class Bucket:
    def __init__(self, bucket_size_b: int, device: str | torch.device, params_info: list[ParamInfo], dtype: torch.dtype, pg: dist.ProcessGroup):
        self.size_mb = bucket_size_b
        self.params_info = params_info
        self.device = device
        self.dtype = dtype
        self.pg = pg

        self.numel = sum(p.numel for p in params_info)  # только фактический буффер коммуницирует (надо проверять профит)
        self.buffer = None  # лениво
        self.param_id_to_info = {p.id: p for p in params_info}

    def get_grad_buffer(self, param_id: int) -> torch.Tensor:
        if self.buffer is None:
            self.buffer = torch.zeros((self.numel,), device=self.device, dtype=self.dtype)

        param_info = self.param_id_to_info[param_id]
        buffer_view = self.buffer[param_info.start : param_info.start + param_info.numel].view(param_info.shape).zero_()

        return buffer_view

    def mark_param_as_ready(self, param_id):
        self.param_id_to_info[param_id].ready = True

        if self._is_bucket_ready():
            assert self.buffer is not None
            dist.all_reduce(self.buffer, group=self.pg)
            self.buffer /= self.pg.size()

            self._mark_bucket_not_ready()

    def _is_bucket_ready(self):
        return all(p.ready is True for p in self.params_info)

    def _mark_bucket_not_ready(self):
        for p in self.params_info:
            p.ready = False
