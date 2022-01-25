import torch

primals_1 = torch.empty([10000, 1, 4], dtype=torch.float32, device='cuda', requires_grad=True)
primals_2 = torch.empty([1, 10000, 4], dtype=torch.float32, device='cuda', requires_grad=True)
tangents_1 = torch.empty([10000, 4], dtype=torch.float32, device='cuda', requires_grad=False)

def backward(primals_1, primals_2, tangents_1):
    unsqueeze = torch.ops.aten.unsqueeze(tangents_1, 0);  tangents_1 = None
    expand = torch.ops.aten.expand(unsqueeze, [10000, 10000, 4]);  unsqueeze = None
    mul_1 = torch.ops.aten.mul(expand, primals_1);  primals_1 = None
    mul_2 = torch.ops.aten.mul(expand, primals_2);  expand = primals_2 = None
    sum_2 = torch.ops.aten.sum(mul_2, [1], True);  mul_2 = None
    sum_3 = torch.ops.aten.sum(mul_1, [0], True);  mul_1 = None
    return [sum_2, sum_3]
    

res = backward(primals_1, primals_2, tangents_1)

