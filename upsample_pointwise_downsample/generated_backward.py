import torch

upsample_nearest1d = torch.empty([32, 512, 2048], dtype=torch.float32, device='cuda', requires_grad=False)
tangents_1 = torch.empty([32, 512, 512], dtype=torch.float32, device='cuda', requires_grad=False)

def backward(upsample_nearest1d, tangents_1):
    upsample_nearest1d_backward = torch.ops.aten.upsample_nearest1d_backward(tangents_1, None, [32, 512, 2048], [0.25]);  tangents_1 = None
    leaky_relu_backward = torch.ops.aten.leaky_relu_backward(upsample_nearest1d_backward, upsample_nearest1d, 0.2, False);  upsample_nearest1d_backward = upsample_nearest1d = None
    upsample_nearest1d_backward_1 = torch.ops.aten.upsample_nearest1d_backward(leaky_relu_backward, None, [32, 512, 512], [4.0]);  leaky_relu_backward = None
    return [upsample_nearest1d_backward_1]
    

res = backward(upsample_nearest1d, tangents_1)

