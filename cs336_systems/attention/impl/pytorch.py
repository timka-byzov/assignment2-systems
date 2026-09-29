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
