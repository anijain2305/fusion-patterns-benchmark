from functorch.compile import aot_function, tvm_compile, clear_compile_cache
from functools import partial


def _save_module(fx_g, args, name=None):
    print("########")
    fx_g.to_folder(name, "Module")
    print(name)
    print(fx_g)
    print("########")
    return fx_g

def save_module(name):
    return partial(_save_module, name=name)

def save_graphs(fn, args):
    fw_compile = save_module("forward")
    bw_compile = save_module("backward")
    print_fn = aot_function(fn, fw_compile, bw_compile)
    print_fn(*args)
    clear_compile_cache()