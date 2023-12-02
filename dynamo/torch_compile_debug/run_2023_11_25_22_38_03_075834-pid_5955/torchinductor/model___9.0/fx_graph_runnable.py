
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.max_autotune = True
torch._inductor.config.triton.cudagraphs = True
torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.2.0a0+gite360f4c
# torch cuda version: None
# torch git version: e360f4c6dd3f0c6a7b949a26cc3bf4fa03a6dd1b


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        abs_1 = torch.ops.aten.abs.default(arg0_1)
        add = torch.ops.aten.add.Tensor(abs_1, 1);  abs_1 = None
        div = torch.ops.aten.div.Tensor(arg0_1, add);  arg0_1 = add = None
        sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
        lt = torch.ops.aten.lt.Scalar(sum_1, 0);  sum_1 = None
        return (div, lt)
        
def load_args(reader):
    buf0 = reader.storage(None, 40)
    reader.tensor(buf0, (10,), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 40)
    reader.tensor(buf1, (10,), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
