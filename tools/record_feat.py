import torch
import torch.nn as nn

# 定义一个示例网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# 初始化网络
model = SimpleNet()

# 存储逐层信息的字典
layer_data = {}

# 定义钩子函数
def forward_hook(module, input, output):
    layer_name = module.__class__.__name__  # 层的名字
    input_data = input[0].detach().clone()  # 输入张量
    output_data = output.detach().clone()  # 输出张量
    weights = None
    if hasattr(module, 'weight') and module.weight is not None:
        weights = module.weight.detach().clone()  # 权重
    layer_data[id(module)] = {
        "layer_name": layer_name,
        "input": input_data,
        "output": output_data,
        "weights": weights
    }

# 注册钩子到每个模块
hooks = []
for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):  # 只对特定模块注册钩子
        hooks.append(module.register_forward_hook(forward_hook))

# 创建输入张量
input_tensor = torch.randn(1, 3, 32, 32)

# 执行前向传播
output = model(input_tensor)

# 打印逐层的输入、输出和权重
print("=== Layer-wise Comparison ===")
for module_id, data in layer_data.items():
    print(f"Layer: {data['layer_name']}")
    print(f"  Input Shape: {data['input'].shape}")
    print(f"  Input Data: {data['input']}")
    print(f"  Output Shape: {data['output'].shape}")
    print(f"  Output Data: {data['output']}")
    if data['weights'] is not None:
        print(f"  Weights Shape: {data['weights'].shape}")
        print(f"  Weights Data: {data['weights']}")
    print("-" * 40)

# 清理钩子
for hook in hooks:
    hook.remove()