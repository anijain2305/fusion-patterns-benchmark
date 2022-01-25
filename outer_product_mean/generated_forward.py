import torch

primals_1 = torch.empty([32, 1000], dtype=torch.float32, device='cuda', requires_grad=True)
primals_2 = torch.empty([32, 2000], dtype=torch.float32, device='cuda', requires_grad=True)

def forward(primals_1, primals_2):
    view = torch.ops.aten.view(primals_1, [32, 1000]);  primals_1 = None
    view_1 = torch.ops.aten.view(primals_2, [32, 2000]);  primals_2 = None
    view_2 = torch.ops.aten.view(view, [32, 1000, 1]);  view = None
    view_3 = torch.ops.aten.view(view_1, [32, 1, 2000]);  view_1 = None
    mul = torch.ops.aten.mul(view_2, view_3)
    mean = torch.ops.aten.mean(mul, [0]);  mul = None
    return [mean, view_2, view_3]
    

res = forward(primals_1, primals_2)

