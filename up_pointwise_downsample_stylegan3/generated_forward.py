import torch

primals_1 = torch.empty([1, 32, 512, 512], dtype=torch.float32, device='cuda', requires_grad=True)
primals_2 = torch.empty([12], dtype=torch.float32, device='cuda', requires_grad=False)
primals_3 = torch.empty([12], dtype=torch.float32, device='cuda', requires_grad=False)
_tensor_constant0 = torch.empty([], dtype=torch.float64, device='cpu', requires_grad=False)
_tensor_constant1 = torch.empty([], dtype=torch.float64, device='cpu', requires_grad=False)

def forward(primals_1, primals_2, primals_3, _tensor_constant0, _tensor_constant1):
    view = torch.ops.aten.view(primals_1, [1, 32, 512, 1, 512, 1]);  primals_1 = None
    constant_pad_nd = torch.ops.aten.constant_pad_nd(view, [0, 1, 0, 0, 0, 1], 0.0);  view = None
    view_1 = torch.ops.aten.view(constant_pad_nd, [1, 32, 1024, 1024]);  constant_pad_nd = None
    mul = torch.ops.aten.mul(primals_2, _tensor_constant0);  primals_2 = _tensor_constant0 = None
    unsqueeze = torch.ops.aten.unsqueeze(mul, 0);  mul = None
    unsqueeze_1 = torch.ops.aten.unsqueeze(unsqueeze, 1);  unsqueeze = None
    repeat = torch.ops.aten.repeat(unsqueeze_1, [32, 1, 1]);  unsqueeze_1 = None
    unsqueeze_2 = torch.ops.aten.unsqueeze(repeat, 2)
    convolution = torch.ops.aten.convolution(view_1, unsqueeze_2, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 32)
    unsqueeze_3 = torch.ops.aten.unsqueeze(repeat, 3);  repeat = None
    convolution_1 = torch.ops.aten.convolution(convolution, unsqueeze_3, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 32)
    leaky_relu = torch.ops.aten.leaky_relu(convolution_1, 0.2)
    mul_1 = torch.ops.aten.mul(leaky_relu, _tensor_constant1);  leaky_relu = _tensor_constant1 = None
    clamp = torch.ops.aten.clamp(mul_1, -256, 256)
    unsqueeze_4 = torch.ops.aten.unsqueeze(primals_3, 0);  primals_3 = None
    unsqueeze_5 = torch.ops.aten.unsqueeze(unsqueeze_4, 1);  unsqueeze_4 = None
    repeat_1 = torch.ops.aten.repeat(unsqueeze_5, [32, 1, 1]);  unsqueeze_5 = None
    unsqueeze_6 = torch.ops.aten.unsqueeze(repeat_1, 2)
    convolution_2 = torch.ops.aten.convolution(clamp, unsqueeze_6, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 32)
    unsqueeze_7 = torch.ops.aten.unsqueeze(repeat_1, 3);  repeat_1 = None
    convolution_3 = torch.ops.aten.convolution(convolution_2, unsqueeze_7, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 32)
    slice_1 = torch.ops.aten.slice(convolution_3, 0, 0, 9223372036854775807);  convolution_3 = None
    slice_2 = torch.ops.aten.slice(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
    slice_3 = torch.ops.aten.slice(slice_2, 2, 0, 9223372036854775807, 2);  slice_2 = None
    slice_4 = torch.ops.aten.slice(slice_3, 3, 0, 9223372036854775807, 2);  slice_3 = None
    return [slice_4, view_1, unsqueeze_2, convolution, unsqueeze_3, convolution_1, mul_1, clamp, unsqueeze_6, convolution_2, unsqueeze_7]
    

res = forward(primals_1, primals_2, primals_3, _tensor_constant0, _tensor_constant1)

