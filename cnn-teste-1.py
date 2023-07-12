import torch
import torch.nn as nn

# Criando um tensor de entrada
input_tensor = torch.randn(1, 3, 32, 32)  # (batch_size, channels, height, width)

# Definindo um módulo de convolução
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# Aplicando a convolução ao tensor de entrada
output_tensor = conv(input_tensor)

# Exibindo as dimensões do tensor de saída
print(output_tensor.size())
