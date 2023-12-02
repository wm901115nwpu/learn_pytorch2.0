class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10]", arg1_1: "f32[10]"):
        # File: /Users/unicorn/workspace/learn_pytorch2.0/dynamo/Frame_Evaluation.py:14, code: b = b * -1
        mul: "f32[10]" = torch.ops.aten.mul.Tensor(arg0_1, -1);  arg0_1 = None
        
        # File: /Users/unicorn/workspace/learn_pytorch2.0/dynamo/Frame_Evaluation.py:15, code: return x * b
        mul_1: "f32[10]" = torch.ops.aten.mul.Tensor(arg1_1, mul);  arg1_1 = mul = None
        return (mul_1,)
        