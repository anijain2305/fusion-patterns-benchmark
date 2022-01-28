import torch

primals_1 = torch.empty([64, 256, 256], dtype=torch.float32, device='cuda', requires_grad=True)
primals_2 = torch.empty([64, 256, 256], dtype=torch.float32, device='cuda', requires_grad=False)

def forward(primals_1, primals_2):
    eq = torch.ops.aten.eq(primals_2, 0);  primals_2 = None
    clone = torch.ops.aten.clone(primals_1, memory_format = 0);  primals_1 = None
    masked_fill_ = torch.ops.aten.masked_fill_(clone, eq, 1e-09);  clone = None
    _softmax = torch.ops.aten._softmax(masked_fill_, -1, False);  masked_fill_ = None
    return [_softmax, eq, _softmax]
    

res = forward(primals_1, primals_2)

