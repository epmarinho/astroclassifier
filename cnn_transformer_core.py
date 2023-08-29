# Author: Eraldo Pereira Marinho, Ph.D
# About: the code is a core module to build a transformer-supported CNN originally design to classify astronomical images
# Creation: Aug 29, 2023
# Usage, import cnn_transformer_core and its components therein

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision
import torchvision.transforms as transforms

nmaxpool = 2
img_width = 384
img_height = 384
img_out_width = img_width // 2**nmaxpool
img_out_height = img_height // 2**nmaxpool

# Transformações de pré-processamento para redimensionar e normalizar as imagens
transform = transforms.Compose([
    transforms.RandomRotation(10),                # Randomly rotate the image by up to 10 degrees
    transforms.RandomHorizontalFlip(),           # Randomly flip the image horizontally
    transforms.Resize((img_width, img_height)),
    transforms.RandomCrop(size=img_height, padding=4),   # Randomly crop the image to size img_height with padding of 4 pixels
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image tensor
])

# Parâmetros de treinamento
num_epochs = 60
batch_size = 32
learning_rate = 0.0001

# Carregar o conjunto de dados de treinamento e teste
train_dataset = torchvision.datasets.ImageFolder(root=r'images/train', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=r'images/tests', transform=transform)

# Criar os dataloaders para facilitar o carregamento dos dados em lotes durante o treinamento
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Parâmetros das redes
num_classes = 4
transformer_layers = 4
cnn_pre_classification = 256
embedding_dimension = cnn_pre_classification
num_heads = 8

# Define a classe do modelo CNN + Transformer
class CNNTransformer(nn.Module):
    # def __init__(self, cnn_model, transformer_layers, num_classes):
    def __init__(self, cnn_model):
        super(CNNTransformer, self).__init__()
        self.cnn_model = cnn_model
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=embedding_dimension, nhead=num_heads),
            num_layers=transformer_layers
        )
        self.fc = nn.Linear(embedding_dimension, num_classes)

    def forward(self, x):
        features = self.cnn_model(x)
        # Assuming features shape: (batch_size, channels, h, w)
        features = features.view(features.size(0), features.size(1), -1)
        features = features.permute(2, 0, 1)  # Reshape for transformer input
        transformed_features = self.transformer(features)
        transformed_features = transformed_features.permute(1, 2, 0)  # Reshape back
        transformed_features = transformed_features.contiguous().view(transformed_features.size(0), -1)
        output = self.fc(transformed_features)
        return output

cnn_n_out_1 = 16
cnn_n_out_2 = 32
cnn_n_out_3 = 64
dense_l_1 = 512
dense_l_2 = 256
dense_l_3 = 128

# Definir a arquitetura da CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(

            # bloco convolutivo 1
            nn.Conv2d(3, cnn_n_out_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 2
            nn.Conv2d(cnn_n_out_1, cnn_n_out_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 3
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 4
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # camadas densas
        expected_flattened_size = cnn_n_out_2 * img_out_width * img_out_height
        self.classifier = nn.Sequential(

            nn.Linear(expected_flattened_size, dense_l_1),
            nn.Dropout(.5),
            nn.ReLU(),

            nn.Linear(dense_l_1, dense_l_2),
            nn.Dropout(.5),
            nn.ReLU(),

            nn.Linear(dense_l_2, cnn_pre_classification),
            nn.Dropout(.5),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Instanciar a CNN + Transformer
# cnn_model = hyperparms.CNN(num_classes=len(train_dataset.classes))
cnn_model = CNN()
# model = CNNTransformer(cnn_model, transformer_layers, num_classes=len(train_dataset.classes))
model = CNNTransformer(cnn_model)
