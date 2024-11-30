import torch
import torch.nn as nn

# 定义网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# 初始化网络
model = SimpleNet()

# 存储算子的字典
operator_counter = {
    "module": {},  # nn.Module 的调用
    "function": {}  # torch.* 函数的调用
}

# 钩子方法统计 nn.Module
def module_hook(module, input, output):
    name = module.__class__.__name__
    if name not in operator_counter["module"]:
        operator_counter["module"][name] = 0
    operator_counter["module"][name] += 1

# 给所有模块注册前向钩子
hooks = []
for module in model.modules():
    hooks.append(module.register_forward_hook(module_hook))

# 使用 autograd 捕获函数调用
from torch.autograd.profiler import record_function

def custom_function_wrapper(func, name):
    def wrapped(*args, **kwargs):
        with record_function(name):  # 记录函数名
            if name not in operator_counter["function"]:
                operator_counter["function"][name] = 0
            operator_counter["function"][name] += 1
            return func(*args, **kwargs)
    return wrapped

# 替换需要统计的 torch 函数
torch.add = custom_function_wrapper(torch.add, "torch.add")
torch.matmul = custom_function_wrapper(torch.matmul, "torch.matmul")

# 输入张量
input_tensor = torch.randn(1, 1, 28, 28)
output = model(input_tensor)  # 前向传播

# 打印统计结果
print("=== Operator Statistics ===")
print("Modules:")
for op, count in operator_counter["module"].items():
    print(f"  {op}: {count}")
print("Functions:")
for op, count in operator_counter["function"].items():
    print(f"  {op}: {count}")

# 清理钩子
for hook in hooks:
    hook.remove()