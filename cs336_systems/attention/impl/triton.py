import torch
import triton
import triton.language as tl
from cs336_basics.nn_utils import softmax
from jaxtyping import Float


class SDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: Float[torch.Tensor, " ... queries d_k"],
        K: Float[torch.Tensor, " ... keys d_k"],
        V: Float[torch.Tensor, " ... keys d_v"],
        is_causal: bool = False,
    ) -> Float[torch.Tensor, "... seq_len d_v"]:

        d_k = Q.new_tensor(float(Q.shape[-1]))
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]

        S = (Q @ K.transpose(-1, -2)) / d_k.sqrt()

        if is_causal:
            S = torch.where(
                torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[None, None, :],
                S,
                -1e6,
            )

        P = softmax(S)
        O = P @ V

        return O


# fmt: off
@triton.jit
def _flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
# fmt: on

    # тут я думаю, что у меня целиком делится N / 16 и D / 16

    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_TILES = tl.cdiv(N_KEYS, K_TILE_SIZE)


    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )


    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Q_tile: tl.tensor = tl.load(Q_block_ptr)
    O_tile: tl.tensor = tl.load(O_block_ptr)

    prev_m = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    for j in range(K_TILES):  # K V inner loop
        # compute P
        K_tile: tl.tensor = tl.load(K_block_ptr)
        S_tile = tl.dot(Q_tile, K_tile.T) * scale
        m = tl.maximum(prev_m, tl.max(S_tile, axis=-1))
        P_tile = tl.exp(S_tile - m[:, None])

        # correction and accumlation
        exp_correction = tl.exp(prev_m - m)
        l = exp_correction * l + tl.sum(P_tile, axis=-1)
        prev_m = m

        # compute O
        V_tile: tl.tensor = tl.load(V_block_ptr)
        O_tile = O_tile * exp_correction[:, None] + tl.dot(P_tile, V_tile)

        # ADVANCE
        V_block_ptr = V_block_ptr.advance((Q_TILE_SIZE, 0))
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))

    O_tile = (1 / l[:, None]) * O_tile

    # WRITE
    tl.store(O_block_ptr, O_tile)
    tl.store(L_block_ptr, prev_m + tl.log(l))


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[torch.Tensor, "b queries d_q"],
        K: Float[torch.Tensor, "b keys d_q"],
        V: Float[torch.Tensor, "b keys d_v"],
        is_causal=False,
    ):
        K_TILE_SIZE = 16
        Q_TILE_SIZE = 16

        D = Q.size(-1)
        SCALE = 1 / D**0.5

        B = Q.size(0)
        N_QUERIES = Q.size(-2)
        N_KEYS = K.size(-2)

        O = torch.zeros((B, N_QUERIES, D), device=Q.device)
        L = torch.zeros((B, N_QUERIES), device=Q.device)

# fmt: off
        _flash_fwd_kernel[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), B)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),            
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            SCALE, D,
            Q_TILE_SIZE, K_TILE_SIZE,
        )
# fmt: on
        ctx.save_for_backward(Q, K, V, O, L)

        return O

    @staticmethod
    def backward(ctx, grad_out):
        # Q, K, V, O, L = ctx.saved_tensors
        # device = Q.device
        # B = Q.size(0)

        # N_QUERIES = Q.size(-2)
        # N_KEYS = K.size(-2)

        # D = Q.size(-1)
        # SCALE = D**0.5

        # K_TILE_SIZE = 16
        # Q_TILE_SIZE = 16

        # dQ_out = torch.zeros((B, N_QUERIES, D), device=device)
        # dK_out = torch.zeros((B, N_KEYS, D), device=device)
        # dV_out = torch.zeros((B, N_KEYS, D), device=device)
        # _flash_attn_bwd(Q, dQ_out, K, dK_out, V, dV_out, O, grad_out, L, N_QUERIES, N_KEYS, Q_TILE_SIZE, K_TILE_SIZE, SCALE)

        # return dQ_out, dK_out, dV_out, None
        pass


f_flashattn = FlashAttention.apply
