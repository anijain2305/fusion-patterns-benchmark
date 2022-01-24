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


# TODO @anijain2305 - Find representative values from the models
S = 32
M = 1000
N = 2000

a = torch.randn(S, M, requires_grad=True, device="cuda")
b = torch.randn(S, N, requires_grad=True, device="cuda")


def f(a, b):
    # Convert to 2D shapes
    a_reshape = a.reshape(-1, a.shape[-1])
    b_reshape = b.reshape(-1, b.shape[-1])

    # Broadcast a and b matrices to prepare for outer_product
    a_broadcast = a_reshape.reshape(a.shape[0], a.shape[1], 1)
    b_broadcast = b_reshape.reshape(b.shape[0], 1, b.shape[1])

    # outer product
    outer = torch.multiply(a_broadcast, b_broadcast)

    # mean
    mean = torch.mean(outer, dim=0)
    return mean


# Save the FX graphs into forward and backward directories
debug.save_graphs(f, (a, b))

# Compile the functions using different backends
target = "cuda -libs=cudnn,cublas"
aot_nvfuser = memory_efficient_fusion(f)
aot_tvm = aot_function(
    f, tvm_compile(target=target), partition_fn=partition_with_recompute_fwd_in_bwd
)

# Check accuracy
debug.check_accuracy(f, (aot_nvfuser, aot_tvm), (a, b))

# Measure perf
baseline_bench = bench.time_with_manual_timer(f, (a, b))
aot_nvfuser_bench = bench.time_with_manual_timer(aot_nvfuser, (a, b), use_nvfuser=True)
aot_tvm_bench = bench.time_with_manual_timer(aot_tvm, (a, b))

# Print the latency
t = PrettyTable(["name", "fwd", "bwd", "total"])
t.add_row(["Eager", *baseline_bench])
t.add_row(["AOT_NvFuser", *aot_nvfuser_bench])
t.add_row(["AOT_TVM", *aot_tvm_bench])
print(t)
