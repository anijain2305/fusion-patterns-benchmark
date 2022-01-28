import torch


def f(scores, mask):
    scores = scores.masked_fill(mask == 0, 1e-9)
    return torch.nn.functional.softmax(scores, dim=-1)


def get_function():
    return f


def get_inputs():
    batch_size = 4
    n_heads = 16
    sequence_len = 256
    shape = (batch_size * n_heads, sequence_len, sequence_len)
    scores = torch.randn(shape, device='cuda', requires_grad=True) 
    mask = torch.rand(shape, device='cuda')
    mask = mask < 0.5
    mask = mask.to(dtype=torch.float32)
    return (scores, mask)
