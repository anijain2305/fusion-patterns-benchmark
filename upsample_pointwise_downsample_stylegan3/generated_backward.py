import torch

view_1 = torch.empty([1, 32, 1024, 1024], dtype=torch.float32, device='cuda', requires_grad=False)
unsqueeze_2 = torch.empty([32, 1, 1, 12], dtype=torch.float32, device='cuda', requires_grad=False)
convolution = torch.empty([1, 32, 1024, 1013], dtype=torch.float32, device='cuda', requires_grad=False)
unsqueeze_3 = torch.empty([32, 1, 12, 1], dtype=torch.float32, device='cuda', requires_grad=False)
convolution_1 = torch.empty([1, 32, 1013, 1013], dtype=torch.float32, device='cuda', requires_grad=False)
mul_1 = torch.empty([1, 32, 1013, 1013], dtype=torch.float32, device='cuda', requires_grad=False)
clamp = torch.empty([1, 32, 1013, 1013], dtype=torch.float32, device='cuda', requires_grad=False)
unsqueeze_6 = torch.empty([32, 1, 1, 12], dtype=torch.float32, device='cuda', requires_grad=False)
convolution_2 = torch.empty([1, 32, 1013, 1002], dtype=torch.float32, device='cuda', requires_grad=False)
unsqueeze_7 = torch.empty([32, 1, 12, 1], dtype=torch.float32, device='cuda', requires_grad=False)
tangents_1 = torch.empty([1, 32, 501, 501], dtype=torch.float32, device='cuda', requires_grad=False)
_tensor_constant2 = torch.empty([1, 32, 1013, 1013], dtype=torch.float32, device='cuda', requires_grad=False)
_tensor_constant1 = torch.empty([], dtype=torch.float64, device='cpu', requires_grad=False)

def backward(view_1, unsqueeze_2, convolution, unsqueeze_3, convolution_1, mul_1, clamp, unsqueeze_6, convolution_2, unsqueeze_7, tangents_1, _tensor_constant2, _tensor_constant1):
    slice_backward = torch.ops.aten.slice_backward(tangents_1, [1, 32, 501, 1002], 3, 0, 9223372036854775807, 2);  tangents_1 = None
    slice_backward_1 = torch.ops.aten.slice_backward(slice_backward, [1, 32, 1002, 1002], 2, 0, 9223372036854775807, 2);  slice_backward = None
    slice_backward_2 = torch.ops.aten.slice_backward(slice_backward_1, [1, 32, 1002, 1002], 1, 0, 9223372036854775807, 1);  slice_backward_1 = None
    slice_backward_3 = torch.ops.aten.slice_backward(slice_backward_2, [1, 32, 1002, 1002], 0, 0, 9223372036854775807, 1);  slice_backward_2 = None
    convolution_backward = torch.ops.aten.convolution_backward(slice_backward_3, convolution_2, unsqueeze_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 32, [True, False, False]);  slice_backward_3 = convolution_2 = unsqueeze_7 = None
    getitem = convolution_backward[0];  convolution_backward = None
    convolution_backward_1 = torch.ops.aten.convolution_backward(getitem, clamp, unsqueeze_6, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 32, [True, False, False]);  getitem = clamp = unsqueeze_6 = None
    getitem_3 = convolution_backward_1[0];  convolution_backward_1 = None
    ge = torch.ops.aten.ge(mul_1, -256)
    le = torch.ops.aten.le(mul_1, 256);  mul_1 = None
    logical_and_ = torch.ops.aten.logical_and_(ge, le);  ge = le = None
    expand = torch.ops.aten.expand(getitem_3, [1, 32, 1013, 1013]);  getitem_3 = None
    expand_1 = torch.ops.aten.expand(logical_and_, [1, 32, 1013, 1013]);  logical_and_ = None
    _s_where = torch.ops.aten._s_where(expand_1, expand, _tensor_constant2);  expand_1 = expand = _tensor_constant2 = None
    mul_2 = torch.ops.aten.mul(_s_where, _tensor_constant1_1);  _s_where = _tensor_constant1_1 = None
    leaky_relu_backward = torch.ops.aten.leaky_relu_backward(mul_2, convolution_1, 0.2, False);  mul_2 = convolution_1 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward(leaky_relu_backward, convolution, unsqueeze_3, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 32, [True, False, False]);  leaky_relu_backward = convolution = unsqueeze_3 = None
    getitem_6 = convolution_backward_2[0];  convolution_backward_2 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward(getitem_6, view_1, unsqueeze_2, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 32, [True, False, False]);  getitem_6 = view_1 = unsqueeze_2 = None
    getitem_9 = convolution_backward_3[0];  convolution_backward_3 = None
    view_2 = torch.ops.aten.view(getitem_9, [1, 32, 512, 2, 512, 2]);  getitem_9 = None
    constant_pad_nd_1 = torch.ops.aten.constant_pad_nd(view_2, [0, -1, 0, 0, 0, -1]);  view_2 = None
    view_3 = torch.ops.aten.view(constant_pad_nd_1, [1, 32, 512, 512]);  constant_pad_nd_1 = None
    return [view_3, None, None, None, None, None, None, None]
    

res = backward(view_1, unsqueeze_2, convolution, unsqueeze_3, convolution_1, mul_1, clamp, unsqueeze_6, convolution_2, unsqueeze_7, tangents_1, _tensor_constant2, _tensor_constant1)

