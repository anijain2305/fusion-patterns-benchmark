# TODO - Code smell, see how to import parent directory modules in a better manner.
import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from functorch.compile import (
    aot_function,
    tvm_compile,
    partition_with_recompute_fwd_in_bwd,
    memory_efficient_fusion,
)
import torch
import debug
import bench
from prettytable import PrettyTable

# TODO - Find representative values from the models
N = 10000
D = 4
a = torch.randn(N, 1, D, requires_grad=True, device="cuda")
b = torch.randn(1, N, D, device="cuda")


def f(a, b):
    return (a * b).sum(dim=0)


debug.save_graphs(f, (a, b))

target = "cuda -libs=cudnn,cublas"
aot_nvfuser = memory_efficient_fusion(f)
aot_tvm = aot_function(
    f, tvm_compile(target=target), partition_fn=partition_with_recompute_fwd_in_bwd
)


baseline_bench = bench.time_with_manual_timer(f, (a, b))
aot_nvfuser_bench = bench.time_with_manual_timer(aot_nvfuser, (a, b), use_nvfuser=True)
aot_tvm_bench = bench.time_with_manual_timer(aot_tvm, (a, b))


t = PrettyTable(["name", "fwd", "bwd", "total"])
t.add_row(["Eager", *baseline_bench])
t.add_row(["AOT_NvFuser", *aot_nvfuser_bench])
t.add_row(["AOT_TVM", *aot_tvm_bench])
print(t)
