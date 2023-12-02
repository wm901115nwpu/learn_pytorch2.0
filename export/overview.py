import torch
from torch.export import export

def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

example_args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: torch.export.ExportedProgram = export(
    torch.compile(f), args=example_args
)
print(exported_program)