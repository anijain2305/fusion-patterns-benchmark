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
import pattern

# TODO - Find representative values from the models
f = pattern.get_function()
args = pattern.get_inputs()

# Save the FX graphs into forward and backward directories
debug.save_graphs(f, args)

# Compile the functions using different backends
target = "cuda -libs=cudnn,cublas"
ts_nvfuser = torch.jit.script(f)
aot_nvfuser = memory_efficient_fusion(f)
aot_tvm = aot_function(
    f, tvm_compile(target=target), partition_fn=partition_with_recompute_fwd_in_bwd
)

# Check accuracy
debug.check_accuracy(f, (ts_nvfuser, aot_nvfuser, aot_tvm), args)

# Measure perf
baseline_bench = bench.time_with_manual_timer(f, args)
ts_nvfuser_bench = bench.time_with_manual_timer(ts_nvfuser, args, use_nvfuser=True)
aot_nvfuser_bench = bench.time_with_manual_timer(aot_nvfuser, args, use_nvfuser=True)
aot_tvm_bench = bench.time_with_manual_timer(aot_tvm, args)

# Print the latency
t = PrettyTable(["name", "fwd", "bwd", "total"])
t.add_row(["Eager", *baseline_bench])
t.add_row(["Torchscript_nvfuser", *ts_nvfuser_bench])
t.add_row(["AOT_nvfuser", *aot_nvfuser_bench])
t.add_row(["AOT_tvm", *aot_tvm_bench])
print(t)
