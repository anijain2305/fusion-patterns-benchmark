import torch
from functorch.compile import aot_function
from functools import partial


def generate_tensor_str(tensor_name, tensor):
    shape = [s for s in tensor.shape]
    device = "'" + str(tensor.device).split(":")[0] + "'"
    requires_grad = str(tensor.requires_grad)
    dtype = tensor.dtype
    tensor_str = f"{tensor_name} = torch.empty({shape}, dtype={dtype}, device={device}, requires_grad={requires_grad})\n"
    return tensor_str


def find_input_names(fx_g):
    def _is_primal(node):
        return node.op == "placeholder"

    primal_inputs = list(filter(_is_primal, fx_g.graph.nodes))
    names = [node.name for node in primal_inputs]
    return names


def _save_module(fx_g, args, name=None):
    assert name == "forward" or name == "backward"
    prologue = str()
    if name == "forward":
        prologue = "import torch\n\n"
    code = fx_g.code
    code = "\n".join([l for l in code.split("\n") if l != ""])

    input_names = find_input_names(fx_g)

    # Generate the tensors for args
    for idx, arg in enumerate(args):
        prologue += generate_tensor_str(input_names[idx], arg)

    # Generate the tensors for buffers
    for buffer_name, buffer in fx_g._buffers.items():
        if buffer is None:
            continue
        buffer_str = generate_tensor_str(buffer_name, buffer)
        prologue += buffer_str
        input_names.append(buffer_name)
        code = "\n".join(
            [l for l in code.split("\n") if f"self.{buffer_name}" not in l]
        )

    # Generate the tensors for params
    for param_name, param in fx_g._parameters.items():
        if param is None:
            continue
        param_str = generate_tensor_str(param_name, param)
        prologue += param_str
        input_names.append(param_name)
        code = "\n".join([l for l in code.split("\n") if f"self.{param_name}" not in l])

    # Setup the function signature to account for buffers and params
    signature = ", ".join(input_names)
    codelines = code.split("\n")
    new_code = str()
    for codeline in codelines:
        if "def forward(" in codeline:
            new_code += f"def {name}({signature}):\n"
        else:
            new_code += codeline + "\n"
    code = new_code

    # Code to run the model
    epilogue = f"res = {name}({signature})"

    # Prepare the final model
    full_model = prologue + "\n" + code + "\n" + epilogue + "\n\n"

    # Write it into a file
    with open(f"generated_{name}.py", "w") as fw:
        fw.write(full_model)

    print(fx_g)
    return fx_g


def save_module(name):
    return partial(_save_module, name=name)


def save_graphs(fn, args):
    fw_compile = save_module("forward")
    bw_compile = save_module("backward")
    print_fn = aot_function(fn, fw_compile, bw_compile)
    print_fn(*args)


def check_accuracy(ref_fn, compile_fns, args):
    ref = ref_fn(*args)
    for compile_fn in compile_fns:
        res = compile_fn(*args)
        assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)
