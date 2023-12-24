""" Code name: cnn-train-transformer-h5-v3.py (main code for training)"""

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
#from cnn_transformer_core_h5_v4 import model
#from cnn_transformer_core_h5_v4 import num_heads
#from cnn_transformer_core_h5_v4 import train_dataloader
#from cnn_transformer_core_h5_v4 import validation_dataloader
#from cnn_transformer_core_h5_v4 import batch_size
from cnn_transformer_core_h5_v4 import class_labels
from cnn_transformer_core_h5_v4 import train_dataset
from cnn_transformer_core_h5_v4 import validation_dataset
from cnn_transformer_core_h5_v4 import CNN
from cnn_transformer_core_h5_v4 import CNNTransformer
import os
import torch.nn.init as init
import numpy as np
# from PIL import Image
import pillow_avif
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {device}")

viz = Visualizer.Visualizer('Astro Classifier', use_incoming_socket=False)

# Define the class weight vector empirically obtained from the last run:
# run after the classes histogram:
galaxies = np.float32(1/188)
globular = np.float32(1/107)
nebulae  = np.float32(1/188)
openclust= np.float32(1/125)
# # run this before to have an actual class histogram
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

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Initialize weights using Xavier uniform initialization
        init.xavier_uniform_(m.weight)

        # Set biases to zero if they exist
        if m.bias is not None:
            init.constant_(m.bias, 0)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Initialize weights using Xavier uniform initialization
        init.xavier_uniform_(m.weight)

        # Set biases to zero if they exist
        if m.bias is not None:
            init.constant_(m.bias, 0)

# Gets the number of classes from the dataset
num_classes = len(class_labels)

num_epochs = 30

learning_rate = 1e-4 # Larger values caused issues

# Training function
update_rate = 2
def train_and_validate(model, dataloader, validation_loader, criterion, optimizer, num_epochs):
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            # # Print or record gradients of intermediate layers
            # for name, param in model.named_parameters():
            #     if param.requires_grad and 'weight' in name:
            #         grdnorm = param.grad.norm().item()
            #         if grdnorm < 0.01:
            #             print(f'Layer: {name}, Grad norm: {grdnorm}')

            running_loss += loss.item() * images.size(0)

        # Update the learning rate based on the scheduler
        scheduler.step()

        if epoch % update_rate == 0:
            validate(model, validation_loader)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
        viz.plot_lines('Batch Loss', epoch_loss)

# The predicted_labels array is used to construct a histogram to reveal how many times each class was predicted during evaluation
predicted_labels = []

# Validation function
def validate(model, dataloader):
    # Initialize variables to keep track of counts
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total = 0
    total_loss = 0
    correct = 0

    model.eval()  # Set the model to evaluation mode

    all_predicted = []
    all_true = []

    with torch.no_grad():
        for images, labels in dataloader:
            # Move the images and labels to the GPU device
            images = images.to(device)
            true_labels = labels.to(device)
            total += true_labels.size(0)

            # Use the trained CNN+Transformer model for the image-set
            outputs = model(images)

            # Compute the Loss function defined in criterion
            loss = criterion(outputs, true_labels)

            # Sum up the computed loss
            total_loss += loss.item()

            # Get the predicted labels
            _, predicted = torch.max(outputs.data, 1)

            # Count the correct ones
            correct += (predicted == true_labels).sum().item()

            predicted_labels.extend(predicted.tolist())

            all_predicted.extend(predicted.tolist())
            all_true.extend(true_labels.tolist())

    validation_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    cm = confusion_matrix(all_true, all_predicted)

    # Compute precision, recall and F1-score for each class
    precision = []
    recall = []
    f1_scores = []

    for i in range(len(cm)):
        true_positive = cm[i, i]
        false_positive = sum(cm[j, i] for j in range(len(cm))) - true_positive
        false_negative = sum(cm[i, j] for j in range(len(cm))) - true_positive

        precision_i = true_positive / (true_positive + false_positive + 1e-8)
        recall_i = true_positive / (true_positive + false_negative + 1e-8)
        f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i + 1e-8)

        precision.append(precision_i)
        recall.append(recall_i)
        f1_scores.append(f1_i)

    print(f'Validation Loss: {validation_loss:.6f}, Validation Accuracy: {accuracy:.2f}%')
    print(f'Precision per class: {precision}')
    print(f'Recall per class: {recall}')
    print(f'F1-score per class: {f1_scores}')
    viz.plot_lines('Validation Loss', validation_loss)
    viz.plot_lines('Validation Accuracy', accuracy)
    viz.plot_lines('Precision', precision)
    viz.plot_lines('Recall', recall)
    viz.plot_lines('F1-scores', f1_scores)

    return accuracy

"""  **** Grid search loop ****  """

# Define the grid for hyperparameters
batch_sizes = [16, 32, 64]
transformer_layers_options = [1, 2, 4, 6]
num_dense_layers_options = [0, 1, 2, 3]
num_heads_options = [2, 4, 8, 16]
embedding_dimensions = [32, 64, 128]

best_accuracy = 0  # Track the best accuracy
best_hyperparameters = None  # Track the best hyperparameters

for batch_size in batch_sizes:
    for transformer_layers in transformer_layers_options:
        for num_dense_layers in num_dense_layers_options:
            for num_heads in num_heads_options:
                for embedding_dimension in embedding_dimensions:

                    fc_out_dim = embedding_dimension
                    dense_dims = [fc_out_dim * 4, fc_out_dim * 2, fc_out_dim] # List of output dimensions for dense layers # The best for unsorted astronomical image classification

                    cnn_out_dim = 2 * dense_dims[0]
                    cnn_out_dims = [cnn_out_dim // 8, cnn_out_dim // 4, cnn_out_dim // 2, cnn_out_dim] # List of output dimensions for convolutional layers

                    print(f'\nEncoder attention embedding dimension = {embedding_dimension}')
                    print(f'Convolutional layers = {cnn_out_dims}')
                    print(f'Full connected laysers = {dense_dims}')
                    print(f'Batch size = {batch_size}')
                    print(f'Transformer layers = {transformer_layers}')
                    print(f'Num dense layers = {num_dense_layers}')
                    print(f'Num heads = {num_heads}')
                    print(f'Embedding dimension = {embedding_dimension}')

                    # Instantiate the CNN + Dense layer + Transformer
                    model = CNNTransformer(CNN(cnn_out_dims, dense_dims),
                                        num_heads=num_heads,
                                        transformer_layers=transformer_layers,
                                        num_dense_layers=num_dense_layers)

                    # Create data loaders
                    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

                    # Restart all the network weights:
                    model.apply(init_weights)

                    # Move the model to the GPU device
                    model.to(device)

                    # Define the loss function and optimizer
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=.5e-6)

                    # Define a scheduler to adjust the learning rate
                    # Here, a StepLR scheduler is used, which reduces the learning rate by a gamma factor after a fixed number of epochs
                    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

                    train_and_validate(model, train_dataloader, validation_dataloader, criterion, optimizer, num_epochs)

                    # Evaluate the model and update best_hyperparameters if it's the best one yet
                    current_accuracy = validate(model, validation_dataloader)  # You might need to define this or modify it to suit your needs
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_hyperparameters = (batch_size, transformer_layers, num_dense_layers, num_heads)
# Grid loop ends here

# Print out the best hyperparameter set and its performance
print(f"Best Hyperparameters: Batch Size={best_hyperparameters[0]}, Transformer Layers={best_hyperparameters[1]}, Dense Layers={best_hyperparameters[2]}, Heads={best_hyperparameters[3]}")
print(f"Best Accuracy: {best_accuracy}")
