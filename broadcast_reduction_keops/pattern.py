import torch


def f(a, b):
    return (a * b).sum(dim=0)


def get_function():
    return f


def get_inputs():
    # TODO - Find representative values from the models
    N = 10000
    D = 4
    a = torch.randn(N, 1, D, requires_grad=True, device="cuda")
    b = torch.randn(1, N, D, device="cuda")
    return (a, b)
