import torch
import numpy as np


def upsampling(x, f, up=1, gain=1):
    batch_size, num_channels, in_height, in_width = x.shape
    upx = upy = up

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)

    # Convolve with the filter.
    f = f.unsqueeze(0).unsqueeze(1).repeat([num_channels, 1] + [1] * f.ndim)
    # f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = torch.nn.functional.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = torch.nn.functional.conv2d(
            input=x, weight=f.unsqueeze(2), groups=num_channels
        )
        x = torch.nn.functional.conv2d(
            input=x, weight=f.unsqueeze(3), groups=num_channels
        )

    return x


def downsampling(x, f, down):
    batch_size, num_channels, in_height, in_width = x.shape
    downx = downy = down

    # Convolve with the filter.
    f = f.unsqueeze(0).unsqueeze(1).repeat([num_channels, 1] + [1] * f.ndim)
    # f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = torch.nn.functional.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = torch.nn.functional.conv2d(
            input=x, weight=f.unsqueeze(2), groups=num_channels
        )
        x = torch.nn.functional.conv2d(
            input=x, weight=f.unsqueeze(3), groups=num_channels
        )

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x


def bias_act(x, alpha, gain, clamp):
    # Evaluate activation function.
    alpha = float(alpha)
    x = torch.nn.functional.leaky_relu(x, alpha)

    # Scale by gain.
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        x = x.clamp(-clamp, clamp)  # pylint: disable=invalid-unary-operand-type
    return x


def f(x, fu, fd):
    """
    Taken from https://github.com/NVlabs/stylegan3/blob/a5a69f58294509598714d1e88c9646c3d7c6ec94/torch_utils/ops/filtered_lrelu.py#L121

    Args:
        x:           Float32/float16/float64 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        fu:          Float32 upsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        fd:          Float32 downsampling FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        b:           Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                     as `x`. The length of vector must must match the channel dimension of `x`.
        up:          Integer upsampling factor (default: 1).
        down:        Integer downsampling factor. (default: 1).
        # padding:     Padding with respect to the upsampled image. Can be a single number
        #              or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
        #              (default: 0).
        gain:        Overall scaling factor for signal magnitude (default: sqrt(2)).
        slope:       Slope on the negative side of leaky ReLU (default: 0.2).
        clamp:       Maximum magnitude for leaky ReLU output (default: None).
        flip_filter: False = convolution, True = correlation (default: False).
    """

    # Ensure that the sample rates here are same as the one in get_inputs
    up_sample_rate = 2
    down_sample_rate = 2

    gain = 1.414
    slope = 0.2
    clamp = 256

    up = up_sample_rate
    down = down_sample_rate
    # Compute using existing ops.
    x = upsampling(x=x, f=fu, up=up, gain=up ** 2)
    x = bias_act(x=x, alpha=slope, gain=gain, clamp=clamp)  # Bias, leaky ReLU, clamp.
    x = downsampling(x=x, f=fd, down=down)
    return x


def get_function():
    return f


def get_inputs():
    up_sample_rate = 2
    down_sample_rate = 2
    batch_size = 1
    num_channels = 32
    in_height = in_width = 512
    fir_filter_size = 6
    up_numtaps = fir_filter_size * up_sample_rate
    down_numtaps = fir_filter_size * down_sample_rate
    x = torch.randn(
        batch_size, num_channels, in_height, in_width, device="cuda", requires_grad=True
    )
    fu = torch.randn(up_numtaps, device="cuda", requires_grad=False)
    fd = torch.randn(down_numtaps, device="cuda", requires_grad=False)

    return x, fu, fd
