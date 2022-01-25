import torch

primals_1 = torch.empty([10000, 1, 4], dtype=torch.float32, device='cuda', requires_grad=True)
primals_2 = torch.empty([1, 10000, 4], dtype=torch.float32, device='cuda', requires_grad=True)

def forward(primals_1, primals_2):
    mul = torch.ops.aten.mul(primals_1, primals_2)
    sum_1 = torch.ops.aten.sum(mul, [0]);  mul = None
    return [sum_1, primals_1, primals_2]
    

res = forward(primals_1, primals_2)

