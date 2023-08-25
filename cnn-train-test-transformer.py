import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision
import torchvision.transforms as transforms
from utils import Visualizer

# ... Restante do seu código ...

# Define a classe do modelo CNN + Transformer
class CNNTransformer(nn.Module):
    def __init__(self, cnn_model, transformer_layers, num_classes):
        super(CNNTransformer, self).__init__()
        self.cnn_model = cnn_model
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=transformer_layers
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.cnn_model(x)
        # Assuming features shape: (batch_size, channels, h, w)
        features = features.view(features.size(0), features.size(1), -1)
        features = features.permute(2, 0, 1)  # Reshape for transformer input
        transformed_features = self.transformer(features)
        transformed_features = transformed_features.permute(1, 2, 0)  # Reshape back
        output = self.fc(transformed_features)
        return output

# ... Restante do seu código ...

# Instanciar a CNN + Transformer
cnn_model = hyperparms.CNN(num_classes=len(train_dataset.classes))
transformer_layers = 2  # Número de camadas do Transformer
model = CNNTransformer(cnn_model, transformer_layers, num_classes=len(train_dataset.classes))

# ... Restante do seu código ...
