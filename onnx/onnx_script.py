import torch
from torch.onnx._internal import jit_utils
import onnxscript
# There are three opset version needed to be aligned
# This is (1) the opset version in ONNX function
from onnxscript.onnx_opset import opset15 as op
opset_version = 15

x = torch.randn(1, 2, 3, 4, requires_grad=True)
model = torch.nn.SELU()

custom_opset = onnxscript.values.Opset(domain="onnx-script", version=1)

@onnxscript.script(custom_opset)
def Selu(X):
    alpha = 1.67326  # auto wrapped as Constants
    gamma = 1.0507
    alphaX = op.CastLike(alpha, X)
    gammaX = op.CastLike(gamma, X)
    neg = gammaX * (alphaX * op.Exp(X) - alphaX)
    pos = gammaX * X
    zero = op.CastLike(0, X)
    return op.Where(X <= zero, neg, pos)

# setType API provides shape/type to ONNX shape/type inference
def custom_selu(g: jit_utils.GraphContext, X):
    return g.onnxscript_op(Selu, X).setType(X.type())

# Register custom symbolic function
# There are three opset version needed to be aligned
# This is (2) the opset version in registry
torch.onnx.register_custom_op_symbolic(
    symbolic_name="aten::selu",
    symbolic_fn=custom_selu,
    opset_version=opset_version,
)

# There are three opset version needed to be aligned
# This is (2) the opset version in exporter
torch.onnx.export(
    model,
    x,
    "model.onnx",
    opset_version=opset_version,
    # only needed if you want to specify an opset version > 1.
    custom_opsets={"onnx-script": 1}
)