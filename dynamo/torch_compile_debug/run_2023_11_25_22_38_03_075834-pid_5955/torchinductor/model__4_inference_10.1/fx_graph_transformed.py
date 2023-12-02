class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10]", arg1_1: "f32[10]"):
        # File: /Users/unicorn/workspace/learn_pytorch2.0/dynamo/Frame_Evaluation.py:15, code: return x * b
        mul: "f32[10]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (mul,)
        