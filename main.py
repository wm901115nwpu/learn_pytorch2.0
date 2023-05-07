from fx_tracer.custom_tracer import custom_symbolic_trace

import torch
import torchvision
import timm

resnet18 = torchvision.models.resnet18()
swin_t = timm.create_model('swin_tiny_patch4_window7_224')
mobilevit_s = timm.create_model('mobilevit_s')

gm = custom_symbolic_trace(mobilevit_s)
print(gm.print_readable())
