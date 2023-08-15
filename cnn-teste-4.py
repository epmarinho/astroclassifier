import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import visdom
from utils import Visualizer

viz = Visualizer.Visualizer('Astro Classifier', use_incoming_socket=False)

nmaxpool = 6
img_width = 512
img_height = 512
img_out_width = img_width // 2**nmaxpool
img_out_height = img_height // 2**nmaxpool

# Definir a arquitetura da CNN
class CNN(nn.Module):
    def __init__(self, num_classes=5):
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
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 5
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # bloco convolutivo 6
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # camadas densas
        self.classifier = nn.Sequential(
            nn.Linear(512 * img_out_width * img_out_height, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Parâmetros de treinamento
num_epochs = 120
batch_size = 32
learning_rate = 0.001

# Transformações de pré-processamento para redimensionar e normalizar as imagens
transform = transforms.Compose([
    transforms.RandomRotation(10),                # Randomly rotate the image by up to 10 degrees
    transforms.RandomHorizontalFlip(),           # Randomly flip the image horizontally
    transforms.Resize((img_width, img_height)),
    # transforms.RandomCrop(size=32, padding=4),   # Randomly crop the image to size 32x32 with padding of 4 pixels
    transforms.ToTensor(),                        # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image tensor
])

# Carregar o conjunto de dados de treinamento e teste
train_dataset = torchvision.datasets.ImageFolder(root=r'images/train', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=r'images/tests', transform=transform)

# Criar os dataloaders para facilitar o carregamento dos dados em lotes durante o treinamento
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instanciar a CNN
model = CNN(num_classes=len(train_dataset.classes))

# Definir a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mover o modelo para o dispositivo GPU
model.to(device)

# Função de treinamento
def train(model, dataloader, test_loader, criterion, optimizer, num_epochs):
    model.train()  # Configurar o modelo para o modo de treinamento

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in dataloader:
            # Mover as imagens e rótulos para o dispositivo GPU
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        if epoch % 10 == 0:
            test(model, test_loader)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        viz.plot_lines('batch loss', epoch_loss)

correct_labels = []

# Função de teste
def test(model, dataloader):
    model.eval()  # Configurar o modelo para o modo de avaliação
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            # Mover as imagens e rótulos para o dispositivo GPU
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Introduzido em 14 de agosto de 2023:
            if (predicted != labels).sum().item():
                correct_labels.extend(predicted.tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    viz.plot_lines('test acuracy', accuracy)

# Mover o modelo para o dispositivo GPU antes do treinamento
model.to(device)

# Treinamento e teste da CNN
train(model, train_loader, test_loader, criterion, optimizer, num_epochs)
test(model, test_loader)

unique_labels = set(correct_labels)
print("unique labels: ", unique_labels)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
indices = [label_to_idx[label] for label in correct_labels]
# print("índices: ", indices)

label_counts = torch.bincount(torch.tensor(indices))

print("label count: ", label_counts)

plt.bar(torch.arange(len(label_counts)), label_counts)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.show()
