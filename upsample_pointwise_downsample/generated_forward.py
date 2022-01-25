import torch

primals_1 = torch.empty([32, 512, 512], dtype=torch.float32, device='cuda', requires_grad=True)

def forward(primals_1):
    upsample_nearest1d = torch.ops.aten.upsample_nearest1d(primals_1, None, [4.0]);  primals_1 = None
    leaky_relu = torch.ops.aten.leaky_relu(upsample_nearest1d, 0.2)
    upsample_nearest1d_1 = torch.ops.aten.upsample_nearest1d(leaky_relu, None, [0.25]);  leaky_relu = None
    return [upsample_nearest1d_1, upsample_nearest1d]
    

res = forward(primals_1)

