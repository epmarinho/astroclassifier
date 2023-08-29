import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
# import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from utils import Visualizer  # You should import your Visualizer module here
import os

nmaxpool = 2
img_width = 384
img_height = 384
img_out_width = img_width // 2**nmaxpool
img_out_height = img_height // 2**nmaxpool
num_classes = 4
transformer_layers = 4

# Transformações de pré-processamento para redimensionar e normalizar as imagens
transform = transforms.Compose([
    transforms.RandomRotation(10),                # Randomly rotate the image by up to 10 degrees
    transforms.RandomHorizontalFlip(),           # Randomly flip the image horizontally
    transforms.Resize((img_width, img_height)),
    transforms.RandomCrop(size=img_height, padding=4),   # Randomly crop the image to size img_height with padding of 4 pixels
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image tensor
])

# Carregar o conjunto de dados de treinamento e teste
# train_dataset = torchvision.datasets.ImageFolder(root=r'images/train', transform=transform)
# test_dataset = torchvision.datasets.ImageFolder(root=r'images/tests', transform=transform)

# Criar os dataloaders para facilitar o carregamento dos dados em lotes durante o treinamento
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        transformed_features = transformed_features.contiguous().view(transformed_features.size(0), -1)
        output = self.fc(transformed_features)
        return output

# Definir a arquitetura da CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(

            # bloco convolutivo 1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
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
        expected_flattened_size = 32 * img_out_width * img_out_height
        self.classifier = nn.Sequential(

            nn.Linear(expected_flattened_size, 512),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(512, 256),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Instanciar a CNN + Transformer
# cnn_model = hyperparms.CNN(num_classes=len(train_dataset.classes))
cnn_model = CNN()
# loaded_model = CNNTransformer(cnn_model, transformer_layers, num_classes=len(train_dataset.classes))
loaded_model = CNNTransformer(cnn_model, transformer_layers, num_classes)

# Load the saved model parameters
saved_model_path = 'trained_cnn_model.pth'
loaded_model.load_state_dict(torch.load(saved_model_path))

# Move the model to the same device as the input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

# Set the model to evaluation mode
loaded_model.eval()

# Transformations for preprocessing the input image
transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image tensor
])

# Replace 'class_labels' with your actual class labels
class_labels = ['galaxies', 'globular clusters', 'nebulae', 'open clusters']
#
# # Load a sample image for inference
# sample_image_path = 'images/eso0024a-1022x1024.jpg'  # Replace with the actual path
# print(sample_image_path)
# sample_image = Image.open(sample_image_path).convert("RGB")
# input_image = transform(sample_image).unsqueeze(0)  # Add an extra dimension for the batch
#
# # Make sure to move the input image to the same device as the model (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_image = input_image.to(device)
#
# # Perform inference
# with torch.no_grad():
#     output = loaded_model(input_image)
#
# # Get the predicted class index
# predicted_class_index = torch.argmax(output).item()
#
# # Print the inferred class
# print(f"Inferred class: {class_labels[predicted_class_index]}")
#
# # Load a sample image for inference
# sample_image_path = 'images/M48_300.jpg'  # Replace with the actual path
# print(sample_image_path)
# sample_image = Image.open(sample_image_path).convert("RGB")
# input_image = transform(sample_image).unsqueeze(0)  # Add an extra dimension for the batch
#
# # Make sure to move the input image to the same device as the model (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_image = input_image.to(device)
#
# # Perform inference
# with torch.no_grad():
#     output = loaded_model(input_image)
#
# # Get the predicted class index
# predicted_class_index = torch.argmax(output).item()
#
# # Replace 'class_labels' with your actual class labels
# class_labels = ['galaxies', 'globular clusters', 'nebulae', 'open clusters', 'others']
#
# # Print the inferred class
# print(f"Inferred class: {class_labels[predicted_class_index]}")

# Path to the validation images directory
validation_dir = 'images/validation'

# Iterate through images in the validation directory
for filename in os.listdir(validation_dir):
    image_path = os.path.join(validation_dir, filename)

    print(f"Processing image: {image_path}")

    # Load and preprocess the image
    sample_image = Image.open(image_path).convert("RGB")
    input_image = transform(sample_image).unsqueeze(0)  # Add an extra dimension for the batch
    input_image = input_image.to(device)  # Move to the same device as the model

    # Perform inference
    with torch.no_grad():
        output = loaded_model(input_image)

    # Get the predicted class index
    predicted_class_index = torch.argmax(output).item()

    # Print the inferred class
    print(f"Inferred class: {class_labels[predicted_class_index]}")
