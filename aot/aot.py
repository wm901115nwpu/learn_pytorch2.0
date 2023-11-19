import torch

def fn(a, b, c, d):
    x = a + b + c + d
    return x.cos().cos()

# Test that it works
a, b, c, d = [torch.randn(2, 4, requires_grad=True) for _ in range(4)]
ref = fn(a, b, c, d)
loss = ref.sum()
loss.backward()

from functorch.compile import aot_function

# The compiler_fn is called after the forward and backward graphs are extracted.
# Here, we just print the code in the compiler_fn. Return of this function is a callable.
def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return fx_module

# Pass on the compiler_fn to the aot_function API
aot_print_fn = aot_function(fn, fw_compiler=compiler_fn, bw_compiler=compiler_fn)

# Run the aot_print_fn once to trigger the compilation and print the graphs
cloned_inputs = [x.clone().detach().requires_grad_(True) for x in (a, b, c, d)]
cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs
res = aot_print_fn(cloned_a, cloned_b, cloned_c, cloned_d)
res.sum().backward()
assert torch.allclose(ref, res)

# AOT Autograd has a suite of already integrated backends. Lets import the NNC compiler backend - ts_compile
from functorch.compile import ts_compile

# Lets compile the forward and backward through ts_compile.
aot_nnc_fn = aot_function(fn, fw_compiler=ts_compile, bw_compiler=ts_compile)

# Correctness checking. Lets clone the input so that we can check grads.
cloned_inputs = [x.clone().detach().requires_grad_(True) for x in (a, b, c, d)]
cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs

res = aot_nnc_fn(*cloned_inputs)
loss = res.sum()
loss.backward()
assert torch.allclose(ref, res)
assert torch.allclose(a.grad, cloned_a.grad)
assert torch.allclose(b.grad, cloned_b.grad)
assert torch.allclose(c.grad, cloned_c.grad)
assert torch.allclose(d.grad, cloned_d.grad)

# Lets write a function to benchmark the forward and backward pass
import time
import statistics


def bench(fn, args, prefix):
    warmup = 10
    iterations = 100

    for _ in range(warmup):
        ref = fn(*args)
        ref.sum().backward()

    fw_latencies = []
    bw_latencies = []
    for _ in range(iterations):
        for arg in args:
            arg.grad = None

        fw_begin = time.perf_counter()
        ref = fn(*args)
        fw_end = time.perf_counter()

        loss = ref.sum()

        bw_begin = time.perf_counter()
        loss.backward()
        bw_end = time.perf_counter()

        fw_latencies.append(fw_end - fw_begin)
        bw_latencies.append(bw_end - bw_begin)

    avg_fw_latency = statistics.mean(fw_latencies) * 10 ** 6
    avg_bw_latency = statistics.mean(bw_latencies) * 10 ** 6
    print(prefix, "Fwd = " + str(avg_fw_latency) + " us", "Bwd = " + str(avg_bw_latency) + " us", sep=', ')

large_inputs = [torch.randn(1024, 2048, requires_grad=True) for _ in range(4)]

# Benchmark the Eager and AOT Autograd functions
bench(fn, large_inputs, "Eager")
bench(aot_nnc_fn, large_inputs, "AOT")


from functorch.compile import min_cut_rematerialization_partition

# Zero out the gradients so we can do a comparison later
a.grad, b.grad, c.grad, d.grad = (None,) * 4

# Lets set up the partitioner. Also set the fwd and bwd compilers to the printer function that we used earlier.
# This will show us how the recomputation has modified the graph.
aot_fn = aot_function(fn, fw_compiler=compiler_fn, bw_compiler=compiler_fn, partition_fn=min_cut_rematerialization_partition)
res = aot_fn(a, b, c, d).sum().backward()

# Lets set up the partitioner and NNC compiler.
aot_recompute_nnc_fn = aot_function(fn, fw_compiler=ts_compile, bw_compiler=ts_compile, partition_fn=min_cut_rematerialization_partition)

# Correctness checking. Lets clone the input so that we can check grads.
cloned_inputs = [x.clone().detach().requires_grad_(True) for x in (a, b, c, d)]
cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs

res = aot_recompute_nnc_fn(*cloned_inputs)
loss = res.sum()
loss.backward()
assert torch.allclose(ref, res)
assert torch.allclose(a.grad, cloned_a.grad)
assert torch.allclose(b.grad, cloned_b.grad)
assert torch.allclose(c.grad, cloned_c.grad)
assert torch.allclose(d.grad, cloned_d.grad)

bench(fn, large_inputs, "Eager")
bench(aot_nnc_fn, large_inputs, "AOT")
bench(aot_recompute_nnc_fn, large_inputs, "AOT_Recomp")