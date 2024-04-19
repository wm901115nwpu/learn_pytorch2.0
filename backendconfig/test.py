import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)

weighted_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float)

def fuse_conv2d_relu(is_qat, conv, relu):
    """Return a fused ConvReLU2d from individual conv and relu modules."""
    return torch.ao.nn.intrinsic.ConvReLU2d(conv, relu)

# For quantizing Linear
linear_config = BackendPatternConfig(torch.nn.Linear) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_root_module(torch.nn.Linear) \
    .set_qat_module(torch.ao.nn.qat.Linear) \
    .set_reference_quantized_module(torch.ao.nn.quantized.reference.Linear)

# For fusing Conv2d + ReLU into ConvReLU2d
conv_relu_config = BackendPatternConfig((torch.nn.Conv2d, torch.nn.ReLU)) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_fused_module(torch.ao.nn.intrinsic.ConvReLU2d) \
    .set_fuser_method(fuse_conv2d_relu)

# For quantizing ConvReLU2d
fused_conv_relu_config = BackendPatternConfig(torch.ao.nn.intrinsic.ConvReLU2d) \
    .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
    .add_dtype_config(weighted_int8_dtype_config) \
    .set_root_module(torch.nn.Conv2d) \
    .set_qat_module(torch.ao.nn.intrinsic.qat.ConvReLU2d) \
    .set_reference_quantized_module(torch.ao.nn.quantized.reference.Conv2d)

backend_config = BackendConfig("my_backend") \
    .set_backend_pattern_config(linear_config) \
    .set_backend_pattern_config(conv_relu_config) \
    .set_backend_pattern_config(fused_conv_relu_config)