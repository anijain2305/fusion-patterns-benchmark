import torch
from functorch.compile import (
    aot_function,
    tvm_compile,
    partition_with_recompute_fwd_in_bwd,
    memory_efficient_fusion,
)
from prettytable import PrettyTable
import argparse
import importlib
import glob
import os

import debug
import bench

OKGREEN = "\033[92m"
FAIL = "\033[91m"
ENDC = "\033[0m"


def run(pattern_name):
    print(f"{OKGREEN}### Starting benchmarking {pattern_name} ###{ENDC}")
    try:
        # Import the pattern module using the pattern name
        module_name = f"{pattern_name}.pattern"
        pattern = importlib.import_module(module_name)
        f = pattern.get_function()
        args = pattern.get_inputs()

        # Save the FX graphs into forward and backward directories
        debug.save_graphs(f, args, path=pattern_name)

        # Compile the functions using different backends
        target = "cuda -libs=cudnn,cublas"
        try:
            ts_nvfuser = torch.jit.script(f)
        except Exception as e:
            print(f"{FAIL}Torchscript failed to script{ENDC}")

        aot_nvfuser = memory_efficient_fusion(f)
        aot_tvm = aot_function(
            f,
            tvm_compile(target=target),
            partition_fn=partition_with_recompute_fwd_in_bwd,
        )

        # Check accuracy
        try:
            debug.check_accuracy(f, (ts_nvfuser, aot_nvfuser, aot_tvm), args)
        except Exception as e:
            print(f"{FAIL}Accuracy test failed - Please look at the error{ENDC}")
            print(e)

        # Measure perf
        baseline_bench = bench.time_with_manual_timer(f, args)
        try:
            ts_nvfuser_bench = bench.time_with_manual_timer(
                ts_nvfuser, args, use_nvfuser=True
            )
        except:
            ts_nvfuser_bench = (-1, -1, -1)
            print(f"{FAIL}Torchscript failed to script{ENDC}")

        aot_nvfuser_bench = bench.time_with_manual_timer(
            aot_nvfuser, args, use_nvfuser=True
        )
        try:
            aot_tvm_bench = bench.time_with_manual_timer(aot_tvm, args)
        except Exception as e:
            aot_tvm_bench = (-1, -1, -1)
            print(f"{FAIL}TVM failed to run because of following exception{ENDC}")
            print(e)

        # Print the latency
        t = PrettyTable(["name", "fwd", "bwd", "total"])
        t.add_row(["Eager", *baseline_bench])
        t.add_row(["Torchscript_nvfuser", *ts_nvfuser_bench])
        t.add_row(["AOT_nvfuser", *aot_nvfuser_bench])
        t.add_row(["AOT_tvm", *aot_tvm_bench])
        print(t)
    except Exception as e:
        print(
            f"{FAIL}FAILED While running pattern {pattern_name} with the following exception{ENDC}"
        )
        print(e)
    print(f"{OKGREEN}### Ended benchmarking {pattern_name} ###\n\n{ENDC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern_name", type=str, default="all")
    args = parser.parse_args()

    pattern_name = args.pattern_name
    if pattern_name == "all":
        all_pattern_names = glob.glob("*/")
        all_pattern_names.remove("__pycache__/")
        all_pattern_names = [r.strip("/") for r in all_pattern_names]
        for pattern_name in all_pattern_names:
            run(pattern_name)
    else:
        run(pattern_name)
