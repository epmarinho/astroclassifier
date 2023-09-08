# Author: Eraldo Pereira Marinho, Ph.D
# About: the code is a core module to build a VGG like CNN with transformer, originally design to classify astronomical images
# Creation: Aug 29, 2023
# Usage, import cnn_transformer_core and its components therein

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

nmaxpool = 3
img_width = 256
img_height = 256
img_out_width = img_width // 2**nmaxpool
img_out_height = img_height // 2**nmaxpool

# Transformações de pré-processamento para redimensionar e normalizar as imagens de treinamento
transform_train = transforms.Compose([
    transforms.RandomRotation(30),                # Randomly rotate the image by up to 30 degrees
    transforms.RandomHorizontalFlip(),           # Randomly flip the image horizontally
    transforms.Resize((img_width, img_height)),
    transforms.RandomCrop(size=(img_width, img_height), padding=4),  # Crop to img_width*img_height with padding 4
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly adjusts brightness, contrast, saturation, and hue.
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image tensor
])

# Transformações de pré-processamento para redimensionar e normalizar as imagens de verificação
transform_test = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image tensor
])

# Carregar o conjunto de dados de treinamento
train_dataset = torchvision.datasets.ImageFolder(root=r'images/train', transform=transform_train)

# Carregar o conjunto de dados de teste
test_dataset = torchvision.datasets.ImageFolder(root=r'images/tests', transform=transform_test)

# Criar os dataloaders para facilitar o carregamento dos dados em lotes durante o treinamento
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

cnn_pre_classification = 512 # este é o número de classes intermediárias como saída do modelo CNN

# Parâmetros do Transformer Encoder
transformer_layers = 2 # número de camadas de atenção do Transformer Encoder
embedding_dimension = cnn_pre_classification # dimensão do espaço de recursos, que é uma dimensão importante para a atenção
num_heads = 16 # número de cabeças de atenção deve ser divisor inteiro de embedding_dimension

# Define a classe do modelo CNN + Transformer
# para classificação de imagens, uma camada Decoder não é necessária. O Encoder do Transformer é usado para extrair recursos úteis da imagem
# e as camadas de classificação subsequentes são usadas para fazer a predição das classes. Este é um design adequado para tarefas de
# classificação de imagem, incluindo a classificação de imagens astronômicas
class CNNTransformer(nn.Module):
    # def __init__(self, cnn_model, transformer_layers, num_classes):
    def __init__(self, cnn_model, num_dense_layers=1):
        super(CNNTransformer, self).__init__()
        self.cnn_model = cnn_model
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=embedding_dimension, nhead=num_heads, activation=F.gelu),
            num_layers=transformer_layers
        )

        # Adding additional dense layers for classification after CNN Transformer
        dense_layers = []
        input_size = embedding_dimension  # Adjust if necessary
        output_size_1 = 128
        output_size_2 = 64

        for _ in range(num_dense_layers):
            dense_layers.append(nn.Linear(input_size, output_size_1))
            dense_layers.append(nn.ReLU())
            input_size = output_size_1

        dense_layers.append(nn.Linear(output_size_1, output_size_2))
        dense_layers.append(nn.ReLU())
        input_size = output_size_2
        self.dense_layers = nn.Sequential(*dense_layers)
        self.fc = nn.Linear(output_size_2, num_classes)

    def forward(self, x):
        features = self.cnn_model(x)
        # Assuming features shape: (batch_size, channels, h, w)
        features = features.view(features.size(0), features.size(1), -1)
        features = features.permute(2, 0, 1)  # Reshape for transformer input
        transformed_features = self.transformer(features)
        transformed_features = transformed_features.permute(1, 2, 0)  # Reshape back
        transformed_features = transformed_features.contiguous().view(transformed_features.size(0), -1)
        transformed_features = self.dense_layers(transformed_features)
        output = self.fc(transformed_features)
        return output

cnn_n_out_1 = 16
cnn_n_out_2 = 32
cnn_n_out_3 = 64
cnn_n_out_4 = 32
dense_l_1 = 128
dense_l_2 = 256
dense_l_3 = 512

# Definir a arquitetura da CNN para extração de features
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(

            # bloco convolutivo 1 - saída maxpool tem metade das dimensões lineares da imagem de entrada redimensionada
            nn.Conv2d(3, cnn_n_out_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_n_out_1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 2 - saída maxpool tem 1/4 das dimensões lineares da imagem de entrada redimensionada
            nn.Conv2d(cnn_n_out_1, cnn_n_out_2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_n_out_2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 3
            nn.Conv2d(cnn_n_out_2, cnn_n_out_3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_n_out_3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 4
            # nn.Conv2d(cnn_n_out_3, cnn_n_out_4, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # camadas densas para pré-classificação a ser usada como ebbeding dimension do Transformer
        dropout = .15
        expected_flattened_size = cnn_n_out_3 * img_out_width * img_out_height
        self.classifier = nn.Sequential(

            nn.Linear(expected_flattened_size, dense_l_1),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            nn.Linear(dense_l_1, dense_l_2),
            # nn.Linear(dense_l_1, cnn_pre_classification),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            nn.Linear(dense_l_2, cnn_pre_classification),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Instanciar a CNN + Transformer
# model = CNNTransformer(cnn_model, transformer_layers, num_classes=len(train_dataset.classes))
num_classes = len(train_dataset.classes)
cnn_model = CNN()
model = CNNTransformer(cnn_model, 2)

