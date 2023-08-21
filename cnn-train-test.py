import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import visdom
from utils import Visualizer

vis = Visualizer.Visualizer('Astro Classifier', use_incoming_socket=False)

# hyperparameters
import hyperparms
nmaxpool = hyperparms.nmaxpool
img_width = hyperparms.img_width
img_height = hyperparms.img_height
img_out_width = hyperparms.img_out_width
img_out_height = hyperparms.img_out_height

# Parâmetros de treinamento
num_epochs = hyperparms.num_epochs
batch_size = hyperparms.batch_size
learning_rate = hyperparms.learning_rate

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
train_dataset = torchvision.datasets.ImageFolder(root=r'images/train', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=r'images/tests', transform=transform)

# Criar os dataloaders para facilitar o carregamento dos dados em lotes durante o treinamento
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instanciar a CNN
model = hyperparms.CNN(num_classes=len(train_dataset.classes))

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

        if epoch % 2 == 0:
            test(model, test_loader)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        vis.plot_lines('batch loss', epoch_loss)

predicted_labels = []

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

            predicted_labels.extend(predicted.tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    vis.plot_lines('test acuracy', accuracy)

# Mover o modelo para o dispositivo GPU antes do treinamento
model.to(device)

# Treinamento e teste da CNN
train(model, train_loader, test_loader, criterion, optimizer, num_epochs)
test(model, test_loader)

# Save the trained model
saved_model_path = 'trained_cnn_model.pth'
torch.save(model.state_dict(), saved_model_path)
# print(f"model.state_dict '{model.state_dict()}'")
print(f"Trained model saved to '{saved_model_path}'")

# plot the histogram for predicted categories - unbalanced histogram means low quality training
unique_labels = set(predicted_labels)
print("unique labels: ", unique_labels)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
indices = [label_to_idx[label] for label in predicted_labels]
# print("índices: ", indices)

label_counts = torch.bincount(torch.tensor(indices))

print("label count: ", label_counts)

plt.bar(torch.arange(len(label_counts)), label_counts)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.show()
