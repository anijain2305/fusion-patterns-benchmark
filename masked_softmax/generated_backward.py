import torch

eq = torch.empty([64, 256, 256], dtype=torch.bool, device='cuda', requires_grad=False)
_softmax = torch.empty([64, 256, 256], dtype=torch.float32, device='cuda', requires_grad=False)
tangents_1 = torch.empty([64, 256, 256], dtype=torch.float32, device='cuda', requires_grad=False)

def backward(eq, _softmax, tangents_1):
    detach = torch.ops.aten.detach(_softmax);  _softmax = None
    detach_1 = torch.ops.aten.detach(detach);  detach = None
    _softmax_backward_data = torch.ops.aten._softmax_backward_data(tangents_1, detach_1, -1, 6);  tangents_1 = detach_1 = None
    clone_1 = torch.ops.aten.clone(_softmax_backward_data);  _softmax_backward_data = None
    masked_fill__1 = torch.ops.aten.masked_fill_(clone_1, eq, 0);  clone_1 = eq = None
    return [masked_fill__1, None]
    

res = backward(eq, _softmax, tangents_1)

