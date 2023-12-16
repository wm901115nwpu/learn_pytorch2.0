import torch 
import torchvision
import torch._dynamo
torch._dynamo.config.suppress_errors = True

model = torchvision.models.resnet18(pretrained=True)
model_compile = torch.compile(model)

torch.onnx.export(model, torch.randn(1, 3, 224, 224), "resnet18.onnx", opset_version=18)
onnx_program = torch.onnx.dynamo_export(model.eval(), torch.randn(1, 3, 224, 224))
onnx_program.save("resnet18_dynamo.onnx")