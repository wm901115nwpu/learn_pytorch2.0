
def guard_2(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['x'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5303454928))) \
        and (___compile_config_hash() == '1ac5f62502e4a517444abfd6d3491bc4') \
        and (___check_tensors(L['b'], L['x'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_3*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_3(*args, **kwargs):
    pass

def compiled_code_2(b, x):
    return __compiled_fn_3(x, b)[0]


# Note: if there is a compiled version below, this function might well not be executed directly. Please check the compiled version if possible.
def __resume_at_38_2(b, x):
    return x * b

def compiled___resume_at_38_2(b, x):
    L = {"b": b, "x": x}
    if guard_2(L):
        return compiled_code_2(b, x)
    # Note: this function might well not be executed directly. It might well be compiled again, i.e. adding one more guards and compiled code.
    return __resume_at_38_2(b, x)

#============ end of __resume_at_38_2 ============#

def guard_1(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['x'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5303454928))) \
        and (___compile_config_hash() == '1ac5f62502e4a517444abfd6d3491bc4') \
        and (___check_tensors(L['b'], L['x'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_4*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_4(*args, **kwargs):
    pass

def compiled_code_1(b, x):
    return __compiled_fn_4(b, x)[0]


# Note: if there is a compiled version below, this function might well not be executed directly. Please check the compiled version if possible.
def __resume_at_30_1(b, x):
    b = b * -1
    return x * b

def compiled___resume_at_30_1(b, x):
    L = {"b": b, "x": x}
    if guard_1(L):
        return compiled_code_1(b, x)
    # Note: this function might well not be executed directly. It might well be compiled again, i.e. adding one more guards and compiled code.
    return __resume_at_30_1(b, x)

#============ end of __resume_at_30_1 ============#

def guard_0(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (hasattr(L['a'], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['b'], '_dynamo_dynamic_indices') == False) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(5303454928))) \
        and (___compile_config_hash() == '1ac5f62502e4a517444abfd6d3491bc4') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['a'], L['b'], tensor_check_names=tensor_check_names))

# Note: please refer to the graph code in __compiled_fn_0*.py.
# Captured Graph: Dynamo generated graph (debuggable when using eager backend).
# Joint graph: joint forward+backward graph from aot autograd.
# Forward graph: forward graph from aot autograd (debuggable when using aot_eager backend).
# Backward graph: backward graph from aot autograd (debuggable when using aot_eager backend).
# AFTER XXX: graph processed by inductor (not debuggable).
def __compiled_fn_0(*args, **kwargs):
    pass

def compiled_code_0(a, b):
    __temp_3 = __compiled_fn_0(a, b)
    x = __temp_3[0]
    if __temp_3[1]:
        return __resume_at_30_1(b, x)
    return __resume_at_38_2(b, x)


# Note: if there is a compiled version below, this function might well not be executed directly. Please check the compiled version if possible.
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def compiled_toy_example(a, b):
    L = {"a": a, "b": b}
    if guard_0(L):
        return compiled_code_0(a, b)
    # Note: this function might well not be executed directly. It might well be compiled again, i.e. adding one more guards and compiled code.
    return toy_example(a, b)

#============ end of toy_example ============#
