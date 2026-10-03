import torch
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


@torch.no_grad
def flash_attn_fwd(
    Q: Float[torch.Tensor, "b queries d_q"],
    K: Float[torch.Tensor, "b keys d_q"],
    V: Float[torch.Tensor, "b keys d_v"],
    O_out: Float[torch.Tensor, "b queries d_v"],
    L_out: Float[torch.Tensor, "b queries"],
    N_QUERIES: int,
    N_KEYS: int,
    Q_TILE_SIZE: int,
    K_TILE_SIZE: int,
    scale: float,
):
    Q_TILES = (N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE
    K_TILES = (N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE

    B = Q.size(0)

    for i in range(Q_TILES):  # Q outer loop
        Q_tile = Q[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :]  # HBM -> SRAM
        O_tile = O_out[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :]  # HBM -> SRAM

        prev_m = torch.zeros((B, Q_TILE_SIZE), device=Q.device)
        l = torch.zeros((B, Q_TILE_SIZE), device=Q.device)

        for j in range(K_TILES):  # K V inner loop
            # compute P
            K_tile = K[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :]  # HBM -> SRAM

            S_tile = Q_tile @ K_tile.transpose(-1, -2) / scale  # SMEM

            tile_m = S_tile.max(dim=-1).values
            m = torch.maximum(prev_m, tile_m)

            P_tile = torch.exp(S_tile - m.unsqueeze(-1))

            exp_correction = torch.exp(prev_m - m)
            prev_m = m

            l = exp_correction * l + P_tile.sum(dim=-1)

            # compute O
            V_tile = V[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :]
            O_tile = O_tile * exp_correction.unsqueeze(-1) + P_tile @ V_tile

        O_tile = (1 / l.unsqueeze(-1)) * O_tile

        # write results
        O_out[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :] = O_tile  # SMEM -> HBM
        L_out[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1)] = prev_m + torch.log(l)  # SMEM -> HBM


@torch.no_grad
def flash_attn_bwd(
    Q: Float[torch.Tensor, "b queries d_q"],
    dQ_out: Float[torch.Tensor, "b queries d_q"],
    K: Float[torch.Tensor, "b keys d_q"],
    dK_out: Float[torch.Tensor, "b keys d_q"],
    V: Float[torch.Tensor, "b keys d_v"],
    dV_out: Float[torch.Tensor, "b keys d_v"],
    O: Float[torch.Tensor, "b queries d_v"],
    dO: Float[torch.Tensor, "b queries d_v"],
    L: Float[torch.Tensor, "b queries"],
    N_QUERIES: int,
    N_KEYS: int,
    Q_TILE_SIZE: int,
    K_TILE_SIZE: int,
    scale: float,
):
    Q_TILES = (N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE
    K_TILES = (N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE

    D = (O * dO).sum(dim=-1)

    for j in range(K_TILES):  # K V outer loop
        K_tile = K[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :]  # HBM -> SRAM
        V_tile = V[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :]  # HBM -> SRAM

        dK_out_tile = dK_out[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :]
        dV_out_tile = dV_out[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :]

        for i in range(Q_TILES):
            Q_tile = Q[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :]  # HBM -> SRAM
            dO_tile = dO[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :]  # HBM -> SRAM
            L_tile = L[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1)]  # HBM -> SRAM
            D_tile = D[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1)]  # HBM -> SRAM

            # check
            S_tile = Q_tile @ K_tile.transpose(-2, -1) / scale
            P_tile = torch.exp(S_tile - L_tile.unsqueeze(-1))

            # 1
            dV_out_tile = dV_out_tile + P_tile.transpose(-2, -1) @ dO_tile  # TARGET

            # 2
            dP_tile = dO_tile @ V_tile.transpose(-2, -1)

            # 3
            dS_tile = P_tile * (dP_tile - D_tile.unsqueeze(-1))

            # 5
            dK_out_tile = dK_out_tile + dS_tile.transpose(-2, -1) @ Q_tile / scale  # TARGET

        dV_out[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :] = dV_out_tile
        dK_out[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :] = dK_out_tile

    # для torch реализации это не нужно
    for i in range(Q_TILES):
        Q_tile = Q[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :]  # HBM -> SRAM
        dO_tile = dO[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :]  # HBM -> SRAM
        dQ_out_tile = dQ_out[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :]
        L_tile = L[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1)]  # HBM -> SRAM
        D_tile = D[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1)]  # HBM -> SRAM

        for j in range(K_TILES):
            K_tile = K[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :]  # HBM -> SRAM
            V_tile = V[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :]  # HBM -> SRAM

            # check
            S_tile = Q_tile @ K_tile.transpose(-2, -1) / scale
            P_tile = torch.exp(S_tile - L_tile.unsqueeze(-1))

            # 2
            dP_tile = dO_tile @ V_tile.transpose(-2, -1)

            # 3
            dS_tile = P_tile * (dP_tile - D_tile.unsqueeze(-1))

            # 4
            dQ_out_tile = dQ_out_tile + dS_tile @ K_tile / scale  # TARGET

        dQ_out[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :] = dQ_out_tile


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
        SCALE = D**0.5

        B = Q.size(0)
        N_QUERIES = Q.size(-2)
        N_KEYS = K.size(-2)

        O_out = torch.zeros((B, N_QUERIES, D), device=Q.device)
        L_out = torch.zeros((B, N_QUERIES), device=Q.device)

        flash_attn_fwd(Q, K, V, O_out, L_out, N_QUERIES, N_KEYS, Q_TILE_SIZE, K_TILE_SIZE, SCALE)

        ctx.save_for_backward(Q, K, V, O_out, L_out)

        return O_out

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, O, L = ctx.saved_tensors
        device = Q.device
        B = Q.size(0)

        N_QUERIES = Q.size(-2)
        N_KEYS = K.size(-2)

        D = Q.size(-1)
        SCALE = D**0.5

        K_TILE_SIZE = 16
        Q_TILE_SIZE = 16

        dQ_out = torch.zeros((B, N_QUERIES, D), device=device)
        dK_out = torch.zeros((B, N_KEYS, D), device=device)
        dV_out = torch.zeros((B, N_KEYS, D), device=device)
        flash_attn_bwd(Q, dQ_out, K, dK_out, V, dV_out, O, grad_out, L, N_QUERIES, N_KEYS, Q_TILE_SIZE, K_TILE_SIZE, SCALE)

        return dQ_out, dK_out, dV_out, None


f_flashattn = FlashAttention.apply
