import torch

view_2 = torch.empty([32, 1000, 1], dtype=torch.float32, device='cuda', requires_grad=True)
view_3 = torch.empty([32, 1, 2000], dtype=torch.float32, device='cuda', requires_grad=True)
tangents_1 = torch.empty([1000, 2000], dtype=torch.float32, device='cuda', requires_grad=False)

def backward(view_2, view_3, tangents_1):
    unsqueeze = torch.ops.aten.unsqueeze(tangents_1, 0);  tangents_1 = None
    expand = torch.ops.aten.expand(unsqueeze, [32, 1000, 2000]);  unsqueeze = None
    div = torch.ops.aten.div(expand, 32);  expand = None
    mul_1 = torch.ops.aten.mul(div, view_2);  view_2 = None
    mul_2 = torch.ops.aten.mul(div, view_3);  div = view_3 = None
    sum_1 = torch.ops.aten.sum(mul_2, [2], True);  mul_2 = None
    sum_2 = torch.ops.aten.sum(mul_1, [1], True);  mul_1 = None
    view_4 = torch.ops.aten.view(sum_2, [32, 2000]);  sum_2 = None
    view_5 = torch.ops.aten.view(sum_1, [32, 1000]);  sum_1 = None
    view_6 = torch.ops.aten.view(view_4, [32, 2000]);  view_4 = None
    view_7 = torch.ops.aten.view(view_5, [32, 1000]);  view_5 = None
    return [view_7, view_6]
    

res = backward(view_2, view_3, tangents_1)

