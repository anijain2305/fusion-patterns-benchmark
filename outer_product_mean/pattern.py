import torch

# TODO @anijain2305 - Find representative values from the models


def f(a, b):
    # Convert to 2D shapes
    a_reshape = a.reshape(-1, a.shape[-1])
    b_reshape = b.reshape(-1, b.shape[-1])

    # Broadcast a and b matrices to prepare for outer_product
    a_broadcast = a_reshape.reshape(a.shape[0], a.shape[1], 1)
    b_broadcast = b_reshape.reshape(b.shape[0], 1, b.shape[1])

    # outer product
    outer = torch.multiply(a_broadcast, b_broadcast)

    # mean
    mean = torch.mean(outer, dim=0)
    return mean


def get_function():
    return f


def get_inputs():
    S = 32
    M = 1000
    N = 2000

    a = torch.randn(S, M, requires_grad=True, device="cuda")
    b = torch.randn(S, N, requires_grad=True, device="cuda")
    return (a, b)
