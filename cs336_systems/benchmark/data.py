import torch


def generate_random_batch(batch_size: int, vocab_size: int = 10_000, max_seq_len: int = 100, device: str | torch.device = "cuda"):

    batch = torch.randint(0, vocab_size, (batch_size, max_seq_len), device=device)
    return batch
