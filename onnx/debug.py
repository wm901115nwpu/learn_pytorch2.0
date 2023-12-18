import torch
import onnx
import onnxruntime
import numpy as np


class DebugOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):
        return x

    @staticmethod
    def symbolic(g, x, name):
        return g.op("my::Debug", x, name_s=name)


debug_apply = DebugOp.apply


class Debugger():
    def __init__(self):
        super().__init__()
        self.torch_value = dict()
        self.onnx_value = dict()
        self.output_debug_name = []

    def debug(self, x, name):
        self.torch_value[name] = x.detach().cpu().numpy()
        return debug_apply(x, name)

    def extract_debug_model(self, input_path, output_path):
        model = onnx.load(input_path)
        inputs = [input.name for input in model.graph.input]
        outputs = []

        for node in model.graph.node:
            if node.op_type == 'Debug':
                debug_name = node.attribute[0].s.decode('ASCII')
                self.output_debug_name.append(debug_name)

                output_name = node.output[0]
                outputs.append(output_name)

                node.op_type = 'Identity'
                node.domain = ''
                del node.attribute[:]
        e = onnx.utils.Extractor(model)
        extracted = e.extract_model(inputs, outputs)
        onnx.save(extracted, output_path)

    def run_debug_model(self, input, debug_model):
        sess = onnxruntime.InferenceSession(debug_model,
                                            providers=['CPUExecutionProvider'])

        onnx_outputs = sess.run(None, input)
        for name, value in zip(self.output_debug_name, onnx_outputs):
            self.onnx_value[name] = value

    def print_debug_result(self):
        for name in self.torch_value.keys():
            if name in self.onnx_value:
                mse = np.mean(self.torch_value[name] - self.onnx_value[name]) ** 2
                print(f"{name} MSE: {mse}")


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.convs1 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1),
                                          torch.nn.Conv2d(3, 3, 3, 1, 1),
                                          torch.nn.Conv2d(3, 3, 3, 1, 1))
        self.convs2 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1),
                                          torch.nn.Conv2d(3, 3, 3, 1, 1))
        self.convs3 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1),
                                          torch.nn.Conv2d(3, 3, 3, 1, 1))
        self.convs4 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1),
                                          torch.nn.Conv2d(3, 3, 3, 1, 1),
                                          torch.nn.Conv2d(3, 3, 3, 1, 1))

    def forward(self, x):
        x = self.convs1(x)
        x = self.convs2(x)
        x = self.convs3(x)
        x = self.convs4(x)
        return x


torch_model = Model()

debugger = Debugger()

from types import MethodType


def new_forward(self, x):
    x = self.convs1(x)
    x = debugger.debug(x, 'x_0')
    x = self.convs2(x)
    x = debugger.debug(x, 'x_1')
    x = self.convs3(x)
    x = debugger.debug(x, 'x_2')
    x = self.convs4(x)
    x = debugger.debug(x, 'x_3')
    return x


torch_model.forward = MethodType(new_forward, torch_model)

dummy_input = torch.randn(1, 3, 10, 10)
torch.onnx.export(torch_model, dummy_input, 'before_debug.onnx', input_names=['input'])

debugger.extract_debug_model('before_debug.onnx', 'after_debug.onnx')

debugger.run_debug_model({'input':dummy_input.numpy()}, 'after_debug.onnx')

debugger.print_debug_result()