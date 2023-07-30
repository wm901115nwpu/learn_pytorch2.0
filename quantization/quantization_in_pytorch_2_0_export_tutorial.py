'''
float_model(Python)                               Input
    \                                              /
     \                                            /
—-------------------------------------------------------
|                    Dynamo Export                     |
—-------------------------------------------------------
                            |
                    FX Graph in ATen     XNNPACKQuantizer,
                            |            or X86InductorQuantizer,
                            |            or <Other Backend Quantizer>
                            |                /
—--------------------------------------------------------
|                 prepare_pt2e_quantizer                |
—--------------------------------------------------------
                            |
                     Calibrate/Train
                            |
—--------------------------------------------------------
|                      convert_pt2e                     |
—--------------------------------------------------------
                            |
                Reference Quantized Model
                            |
—--------------------------------------------------------
|                        Lowering                       |
—--------------------------------------------------------
                            |
        Executorch, or Inductor, or <Other Backends>
'''

import torch
import copy
import torch._dynamo as torchdynamo
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
import torch.ao.quantization.pt2e.quantizer.xnnpack_quantizer as xq

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)

    def forward(self, x):
        return self.linear(x)

example_inputs = (torch.randn(1, 5),)
model = M().eval()

# Step 1: Trace the model into an FX graph of flattened ATen operators
exported_graph_module, guards = torchdynamo.export(
    model,
    *copy.deepcopy(example_inputs),
    aten_graph=True,
)

# Step 2: Insert observers or fake quantize modules
quantizer = xq.XNNPACKQuantizer()
operator_config = xq.get_symmetric_quantization_config(is_per_channel=True)
quantizer.set_global(operator_config)
prepared_graph_module = prepare_pt2e(exported_graph_module, quantizer)

# Step 3: Quantize the model
convered_graph_module = convert_pt2e(prepared_graph_module)

# Step 4: Lower Reference Quantized Model into the backend


# Annotate common operator patterns
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
import itertools
import operator
gm = exported_graph_module
add_partitions = get_source_partitions(gm.graph, [operator.add, torch.add])
add_partitions = list(itertools.chain(*add_partitions.values()))
for add_partition in add_partitions:
    add_node = add_partition.output_nodes[0]