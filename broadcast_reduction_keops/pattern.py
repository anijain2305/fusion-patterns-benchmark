from functorch.compile import aot_function, tvm_compile, clear_compile_cache
import torch
import bench
import debug


# TODO - Find representative values from the models
N = 10000
D = 4
a = torch.randn(N, 1, D, requires_grad=True, device='cuda')
b = torch.randn(1, N, D, device='cuda')


def f(a, b):
    return (a * b).sum(dim=0)

debug.save_graphs(f, (a, b))

target = 'cuda -libs=cudnn,cublas'
compiled_f = aot_function(f, tvm_compile(target=target))

baseline = bench.time_with_manual_timer(f, (a, b))
aot = bench.time_with_manual_timer(compiled_f, (a, b))
print("Name", "fwd", "bwd", "total", sep='\t')
print("Eager", *baseline, sep='\t')
print("Aot", *aot, sep='\t')
