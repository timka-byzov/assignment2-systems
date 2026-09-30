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


def flash_attn_fwd(
    Q: Float[torch.Tensor, "b queries d_q"],
    K: Float[torch.Tensor, "b keys d_q"],
    V: Float[torch.Tensor, "b keys d_v"],
    O: Float[torch.Tensor, "b queries d_v"],
    L: Float[torch.Tensor, "b queries"],
    N_QUERIES: int,
    N_KEYS: int,
    Q_TILE_SIZE: int,
    K_TILE_SIZE: int,
    scale: float,
):
    Q_TILES = (N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE
    K_TILES = (N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE

    B = Q.size(0)

    for i in range(Q_TILES):
        Q_tile = Q[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :]  # [B_q, d]
        O_tile = O[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :]  # [B_q, d]

        prev_m = torch.zeros((B, Q_TILE_SIZE), device=Q.device)
        l = torch.zeros((B, Q_TILE_SIZE), device=Q.device)

        for j in range(K_TILES):
            # compute P
            K_tile = K[:, K_TILE_SIZE * j : K_TILE_SIZE * (j + 1), :]  # [B_k, d]

            S_tile = Q_tile @ K_tile.transpose(-1, -2) / scale  # [B_q, B_k]

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
        O[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1), :] = O_tile
        L[:, Q_TILE_SIZE * i : Q_TILE_SIZE * (i + 1)] = prev_m + torch.log(l)


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

        O = torch.zeros((B, N_QUERIES, D), device=Q.device)
        L = torch.zeros((B, N_QUERIES), device=Q.device)

        flash_attn_fwd(Q, K, V, O, L, N_QUERIES, N_KEYS, Q_TILE_SIZE, K_TILE_SIZE, SCALE)

        ctx.save_for_backward(Q, K, V, O, L)

        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError()


f_flashattn = FlashAttention.apply
