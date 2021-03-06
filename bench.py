import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.benchmark import Timer
import time


def profile_cuda_kernels(fn, args):
    warmup = 50
    old_args = args[:]
    n_repeats = 1
    n_layers = 1
    ref = fn(*old_args)
    gO = torch.rand_like(ref)
    for _ in range(0, warmup // n_layers):
        args = list(old_args[:])
        ref = fn(*args)
        ref.backward(gO)

    torch.cuda.synchronize()

    # Forward profile
    def fwd_run():
        for _ in range(0, n_repeats // n_layers):
            args = list(old_args[:])
            for arg in args:
                arg.grad = None
            ref = fn(*args)

    print(f"###### Forward profile starts #####")
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("baseline"):
            fwd_run()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print(f"###### Forward profile ends #####")

    # Backward profile
    def bwd_run():
        for _ in range(0, n_repeats // n_layers):
            args = list(old_args[:])
            for arg in args:
                arg.grad = None
            ref = fn(*args)

            print(f"###### Backward profile starts #####")
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CUDA], record_shapes=True
            ) as prof:
                with record_function("baseline"):
                    ref.backward(gO)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            torch.cuda.synchronize()
            print(f"###### Backward profile ends #####")

    bwd_run()


def time_with_torch_timer(fn, args, kwargs={}):
    ref = fn(*args, **kwargs)
    gO = torch.rand_like(ref)
    env = {"args": args, "gO": gO, "kwargs": kwargs, "fn": fn}
    grad_none = {"for x in args: x.grad=None"}
    fn_call = "fn(*args, **kwargs)"
    # Measure end-to-end fwd time
    timer = Timer(stmt=f"{fn_call}", globals=env)
    fwd_latency = round(timer.timeit(1000).mean * 10 ** 6, 3)
    timer_blocked = timer.blocked_autorange()
    print(f"Forward = {fwd_latency}")

    # Measure end-to-end fwd bwd
    timer = Timer(
        stmt=f"{grad_none}; fwd = {fn_call}; fwd.backward(gO)",
        globals=env,
    )
    fwd_bwd_latency = round(timer.timeit(1000).mean * 10 ** 6, 3)
    timer_blocked = timer.blocked_autorange()
    # print(f"Forward + sum + Backward = {fwd_sum_bwd_latency}")

    bwd_latency = round(fwd_bwd_latency - fwd_latency, 3)
    print(f"Backward = {bwd_latency}")
    return fwd_latency, bwd_latency


def _time_with_manual_timer(fn, args, warmup=50, repeats=1000):
    old_args = args[:]
    ref = fn(*old_args)
    gO = torch.rand_like(ref)
    for _ in range(0, warmup):
        args = list(old_args[:])

        for arg in args:
            arg.grad = None
        ref = fn(*args)
        ref.backward(gO)

    torch.cuda.synchronize()

    fwd_times = []
    bwd_times = []
    for _ in range(0, repeats):
        args = list(old_args[:])
        for arg in args:
            arg.grad = None
        fwd_start = time.time()
        ref = fn(*args)
        torch.cuda.synchronize()
        fwd_end = time.time()

        bwd_start = time.time()
        ref.backward(gO)
        torch.cuda.synchronize()
        bwd_end = time.time()

        fwd_times.append(fwd_end - fwd_start)
        bwd_times.append(bwd_end - bwd_start)
    avg_fwd = round(sum(fwd_times) / repeats * 10 ** 6, 2)
    avg_bwd = round(sum(bwd_times) / repeats * 10 ** 6, 2)
    avg_total = round(avg_fwd + avg_bwd, 2)

    # print(f"Forward = {avg_fwd}")
    # print(f"Backward = {avg_bwd}")
    return avg_fwd, avg_bwd, avg_total


def time_with_manual_timer(fn, args, warmup=50, repeats=1000, use_nvfuser=False):
    if use_nvfuser:
        with torch.jit.fuser("fuser2"):
            return _time_with_manual_timer(fn, args, warmup, repeats)
    return _time_with_manual_timer(fn, args, warmup, repeats)
