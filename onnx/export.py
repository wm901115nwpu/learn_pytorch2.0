import torch 
import torchvision
import torch._dynamo
torch._dynamo.config.suppress_errors = True

model = torchvision.models.convnext_tiny(pretrained=True)
model_compile = torch.compile(model)

# torch.onnx.export(model, torch.randn(1, 3, 224, 224), "convnext_tiny.onnx", opset_version=18)
print(type(model_compile))
torch.onnx.export(model_compile.eval(), torch.randn(1, 3, 224, 224), "convnext_tiny_compile.onnx")
# onnx_program = torch.onnx.dynamo_export(model.eval(), torch.randn(1, 3, 224, 224))
# onnx_program.save("convnext_tiny_dynamo.onnx")