import torch


def f(x):
    x = torch.nn.functional.interpolate(x, scale_factor=4.0)
    x = torch.nn.functional.leaky_relu(x, 0.2)
    x = torch.nn.functional.interpolate(x, scale_factor=0.25)
    return x


def get_function():
    return f


def get_inputs():
    batch_size = 1
    num_channels = 32
    in_height = in_width = 512
    x = torch.randn(
        batch_size, num_channels, in_height, in_width, device="cuda", requires_grad=True
    )

    return x
