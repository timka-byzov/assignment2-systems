import torch
import triton
import triton.language as tl
from jaxtyping import Float


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

    prev_m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
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
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))

    O_tile = (1 / l[:, None]) * O_tile

    # WRITE
    tl.store(O_block_ptr, O_tile)
    tl.store(L_block_ptr, prev_m + tl.log(l))



@triton.jit
def _flash_bwd_kernel_kv(
    Q_ptr, K_ptr, V_ptr,
    dK_ptr, dV_ptr, dO_ptr,
    L_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_vb, stride_vk, stride_vd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    kv_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_TILES = tl.cdiv(N_QUERIES, Q_TILE_SIZE)


# OUTER CYCLE BLOCKS
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(kv_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(kv_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )


    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(kv_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(kv_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )


# INNER CYCLE BLOCKS

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    K_tile: tl.tensor = tl.load(K_block_ptr)
    V_tile: tl.tensor = tl.load(V_block_ptr)    
    dK_tile: tl.tensor = tl.load(dK_block_ptr)
    dV_tile: tl.tensor = tl.load(dV_block_ptr)


    for j in range(Q_TILES):  # Q inner loop
        Q_tile: tl.tensor = tl.load(Q_block_ptr)
        dO_tile: tl.tensor = tl.load(dO_block_ptr)
        L_tile: tl.tensor = tl.load(L_block_ptr)
        D_tile: tl.tensor = tl.load(D_block_ptr)

        # CHECK
        S_tile = tl.dot(Q_tile, K_tile.T) * scale
        P_tile = tl.exp(S_tile - L_tile[:, None])

        # 1
        dV_tile = dV_tile + tl.dot(P_tile.T, dO_tile)   # TARGET

        # 2
        dP_tile = tl.dot(dO_tile, V_tile.T)

        # 3
        dS_tile = P_tile * (dP_tile - D_tile[:, None])

        # 5
        dK_tile = dK_tile + tl.dot(dS_tile.T, Q_tile) * scale  # TARGET

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dK_tile)
    tl.store(dV_block_ptr, dV_tile)


@triton.jit
def _flash_bwd_kernel_q(
    Q_ptr, K_ptr, V_ptr,
    dQ_ptr, dO_ptr,
    L_ptr, D_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_dqb, stride_dqq, stride_dqd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):

    q_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_TILES = tl.cdiv(N_KEYS, K_TILE_SIZE)
    
    # OUTER CYCLE BLOCKS
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(q_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # INNER CYCLE BLOCKS
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


    Q_tile: tl.tensor = tl.load(Q_block_ptr)
    dO_tile: tl.tensor = tl.load(dO_block_ptr)
    L_tile: tl.tensor = tl.load(L_block_ptr)
    D_tile: tl.tensor = tl.load(D_block_ptr)
    dQ_tile: tl.tensor = tl.load(dQ_block_ptr)
        

    for j in range(K_TILES):
        K_tile: tl.tensor = tl.load(K_block_ptr)
        V_tile: tl.tensor = tl.load(V_block_ptr)  

        # CHECK
        S_tile = tl.dot(Q_tile, K_tile.T) * scale
        P_tile = tl.exp(S_tile - L_tile[:, None])

        # 2
        dP_tile = tl.dot(dO_tile, V_tile.T)

        # 3
        dS_tile = P_tile * (dP_tile - D_tile[:, None])

        # 4
        dQ_tile = dQ_tile + tl.dot(dS_tile, K_tile) * scale  # TARGET

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_tile)



# чтобы не материализовывать O * dO
@torch.compile
def _get_D(O: Float[torch.Tensor, "b queries d_q"], dO: Float[torch.Tensor, "b queries d_q"]) -> Float[torch.Tensor, "b queries"]:
    return (O * dO).sum(dim=-1)

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

        O = torch.zeros((B, N_QUERIES, D), device=Q.device, dtype=Q.dtype)
        L = torch.zeros((B, N_QUERIES), device=Q.device, dtype=torch.float32)

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
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        device = Q.device
        B = Q.size(0)

        N_QUERIES = Q.size(-2)
        N_KEYS = K.size(-2)

        D = Q.size(-1)
        SCALE = 1 / D**0.5

        K_TILE_SIZE = 16
        Q_TILE_SIZE = 16

        dQ_out = torch.zeros((B, N_QUERIES, D), device=device, dtype=Q.dtype)
        dK_out = torch.zeros((B, N_KEYS, D), device=device, dtype=K.dtype)
        dV_out = torch.zeros((B, N_KEYS, D), device=device, dtype=V.dtype)

        row_D = _get_D(O, dO)
# fmt: off
        _flash_bwd_kernel_kv[(triton.cdiv(N_KEYS, K_TILE_SIZE), B)](
            Q, K, V,
            dK_out, dV_out, dO,
            L, row_D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            dK_out.stride(0), dK_out.stride(1), dK_out.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dV_out.stride(0), dV_out.stride(1), dV_out.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            row_D.stride(0), row_D.stride(1),
            N_QUERIES, N_KEYS,
            SCALE, D,
            Q_TILE_SIZE, K_TILE_SIZE,
        )

        _flash_bwd_kernel_q[(triton.cdiv(N_QUERIES, Q_TILE_SIZE), B)](
            Q, K, V,
            dQ_out, dO,
            L, row_D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            dQ_out.stride(0), dQ_out.stride(1), dQ_out.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            row_D.stride(0), row_D.stride(1),
            N_QUERIES, N_KEYS,
            SCALE, D,
            Q_TILE_SIZE, K_TILE_SIZE,
        )
# fmt: on

        return dQ_out, dK_out, dV_out, None


f_flashattn = FlashAttention.apply
