# Author: Eraldo Pereira Marinho, Ph.D
# About: The code imports cnn_transformer_core to allow Transformer+CNN to classify astronomical images
# Creation: Jul 12, 2023

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
# import torchvision.transforms as transforms
import visdom
from utils import Visualizer
from cnn_transformer_core_h5_v3 import model
# from cnn_transformer_core_h5_v3 import learning_rate
from cnn_transformer_core_h5_v3 import train_loader
from cnn_transformer_core_h5_v3 import validation_loader
# from cnn_transformer_core_h5_v3 import num_epochs
# from cnn_transformer_core_h5_v3 import Swish
import os
import torch.nn.init as init
import numpy as np
# from PIL import Image
import pillow_avif

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {device}")

viz = Visualizer.Visualizer('Astro Classifier', use_incoming_socket=False)

# Define the class weight vector empirically obtained from the last run:
galaxies = np.float32(1/185)
globular = np.float32(1/118)
nebulae  = np.float32(1/196)
openclust= np.float32(1/121)
# galaxies = np.float32(1)
# globular = np.float32(1)
# nebulae  = np.float32(1)
# openclust= np.float32(1)
norm_denominator=galaxies + globular + nebulae + openclust
weight_class_0=galaxies/norm_denominator
weight_class_1=globular/norm_denominator
weight_class_2=nebulae/norm_denominator
weight_class_3=openclust/norm_denominator
# Instantiate the class weight tensor
class_weights = torch.tensor([weight_class_0, weight_class_1, weight_class_2, weight_class_3])
print(f"Class weights = {class_weights}")
# Weights tensor must be converted to the adopted device
class_weights = class_weights.to(device)

# Training parameters
num_epochs = 40
initial_learning_rate = 1e-4 # Larger values caused issues
# Define the loss function and optimizer
weight_decay = 1e-7
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=weight_decay)
# Define a scheduler to adjust the learning rate
# Here, a StepLR scheduler is used, which reduces the learning rate by a gamma factor after a fixed number of epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Check if the pretrained file exists
model_checkpoint = "trained_cnn_model.pth"
if os.path.exists(model_checkpoint):
    # Load pretrained weights
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint)
    print("Pretrained weights loaded successfully.")
else:
    print("No pretrained weights file found. Initializing with PyTorch default weights.")
    # He initialization in PyTorch
    # Access all the linear layers (fully connected)
    # Ensure the model contains only layers that should be initialized with He
    # for layer in model.children():
    #     if isinstance(layer, nn.Linear):
    #         init.kaiming_normal_(layer.weight)

# Check the loaded weights
# print(model.state_dict())

# Move the model to the GPU device
model.to(device)

# Training function
def train(model, dataloader, validation_loader, criterion, optimizer, num_epochs):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in dataloader:
            # Move the images and labels to the GPU device
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Print or record gradients of intermediate layers
            # for name, param in model.named_parameters():
            #     if param.requires_grad and 'weight' in name:
            #         grdnorm = param.grad.norm().item()
            #         if grdnorm < 0.01:
            #             print(f'Layer: {name}, Grad norm: {grdnorm}')

            running_loss += loss.item() * images.size(0)

        # Update the learning rate based on the scheduler
        scheduler.step()

        if epoch % 10 == 0:
            validate(model, validation_loader)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
        viz.plot_lines('batch loss', epoch_loss)

# The predicted_labels array is used to construct a histogram to reveal how many times each class was predicted during evaluation
predicted_labels = []

# Validation function
def validate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            # Move the images and labels to the GPU device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            predicted_labels.extend(predicted.tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    viz.plot_lines('validation accuracy', accuracy)

# Move the model to the GPU device before training
model.to(device)

# Train and validate the CNN
train(model, train_loader, validation_loader, criterion, optimizer, num_epochs)
validate(model, validation_loader)

# Save the trained weights
saved_model_path = 'trained_cnn_model.pth'
torch.save(model.state_dict(), saved_model_path)
# print(f"model.state_dict '{model.state_dict()}'")
print(f"Trained model saved to '{saved_model_path}'")
# # Load pretrained weights
# checkpoint = torch.load(model_checkpoint)
# model.load_state_dict(checkpoint)
# print("Pretrained weights loaded successfully.")
# # Check the loaded weights
# torch.save(model.state_dict(), 'pesos_lidos.pth')
# torch.save(model.state_dict(), 'trained_cnn_model.pth')

# Plot the histogram for predicted categories - an unbalanced histogram indicates low-quality training
unique_labels = set(predicted_labels)
print("Unique labels: ", unique_labels)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
indices = [label_to_idx[label] for label in predicted_labels]
# print("Indices: ", indices)

label_counts = torch.bincount(torch.tensor(indices))

print("Label count: ", label_counts)

plt.bar(torch.arange(len(label_counts)), label_counts)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.show()
