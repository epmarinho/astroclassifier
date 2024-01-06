""" Code name: cnn-train-transformer-h5-v3.py (main code for training)"""

# Author: Eraldo Pereira Marinho, Ph.D
# About: The code imports cnn_transformer_core to allow Transformer+CNN to classify astronomical images
# Creation: Jul 12, 2023
#
# This is a benchmark script to find out the optimal hyperparameters.

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import visdom
from utils import Visualizer
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
vis = visdom.Visdom()

# Define the class weight vector empirically obtained from the last run:
# run after the classes histogram:
galaxies = np.float32(1/1582)
globular = np.float32(1/498)
nebulae  = np.float32(1/734)
openclust= np.float32(1/450)
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

# Gets the number of classes from the dataset
num_classes = len(class_labels)

num_epochs = 100

# These are basically my earling stopping proposed in previously unpublished works
class EarlyStoppingBatch:
    def __init__(self, patience,  threshold=.005):
        self.history = []
        self.patience = patience
        self.threshold = threshold

    def update_history(self, new_loss):
        # Update the history array with the new loss value
        self.history.append(new_loss)
        if len(self.history) > self.patience:
            self.history.pop(0) # Discard the earliest one

    def should_stop(self):
        # Check if the minimum loss in the history is repeated or becomes smaller
        if len(self.history) < self.patience:
            return False  # Not enough data to decide
        return self.history[-1] <= min(self.history[:-1]) + self.threshold

class EarlyStoppingValLoss:
    def __init__(self, patience,  threshold=.005):
        self.history = []
        self.patience = patience
        self.threshold = threshold

    def update_history(self, new_loss):
        # Update the history array with the new loss value
        self.history.append(new_loss)
        # Keep only the most recent 'remembrance' elements
        if len(self.history) > self.patience:
            self.history.pop(0) # Discard the earliest one

    def should_stop(self):
        # Check if we have enough data to make a decision
        if len(self.history) < self.patience:
            return False  # Not enough data to decide

        # Check the best loss so far
        min_loss = min(self.history[:-1])

        # Check if the loss has not improved significantly for 'patience' epochs
        plateau_count = sum(1 for x in self.history[-self.patience:] if min_loss - self.threshold <= x <= min_loss + self.threshold)

        # If the loss has been on a plateau for 'patience' consecutive epochs, stop
        return plateau_count >= self.patience

class EarlyStoppingAccuracy:
    def __init__(self, patience,  threshold=.0005):
        self.history = []
        self.patience = patience
        self.threshold = threshold

    def update_history(self, new_accuracy):
        # Update the history array with the new loss value
        self.history.append(new_accuracy)
        if len(self.history) > self.patience:
            self.history.pop(0) # Discard the earliest one

    def should_stop(self):
        """
        Determine if training should be stopped based on validation loss.

        Returns:
        - Boolean, True if training should be stopped, False otherwise.
        """
        if len(self.history) < self.patience:
            return False  # Not enough data to decide, continue training

        # Check the best loss so far
        max_accuracy = max(self.history[:-1])

        # Count how many recent losses are within the threshold of the best loss
        plateau_count = sum(1 for x in self.history[-self.patience:] if max_accuracy - self.threshold <= x <= max_accuracy + self.threshold)

        # Stop if the loss hasn't improved for 'patience' consecutive epochs
        return plateau_count >= self.patience

learning_rate = 1e-4 # Larger values caused issues

# Training function
def train_and_validate(model, dataloader, validation_loader, criterion, optimizer, max_norm=2):
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            running_loss += loss.item() * images.size(0)

        # Update the learning rate based on the scheduler
        scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}\n')
        viz.plot_lines('Batch Loss', epoch_loss)

        early_stopping_batch.update_history(epoch_loss)
        if early_stopping_batch.should_stop():
            print(f"\nEarly stopping triggered at epoch {epoch + 1} for batch loss = {epoch_loss}\n")
            break

        validation_loss, accuracy = validate(model, validation_loader)

        scheduler_by_valloss.step(validation_loss)
        early_stopping_valloss.update_history(validation_loss)
        if early_stopping_valloss.should_stop():
            print(f"\nEarly stopping triggered at epoch {epoch + 1} for validation loss = {validation_loss}\n")
            break

        scheduler_by_accuracy.step(accuracy)
        early_stopping_accuracy.update_history(accuracy)
        if early_stopping_accuracy.should_stop():
            print(f"\nEarly stopping triggered at epoch {epoch + 1} for validation accuracy = {accuracy:.2f}%\n")
            break

# The predicted_labels array is used to construct a histogram to reveal how many times each class was predicted during evaluation
predicted_labels = []

# Validation function

# The predicted_labels array is used to construct a histogram to reveal how many times each class was predicted during evaluation
predicted_labels = []

from sklearn.metrics import confusion_matrix

# Validation function
def validate(model, dataloader):

    model.eval()  # Set the model to evaluation mode

    # Initialize variables to keep track of counts
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total = 0
    total_loss = 0
    correct = 0

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

    #early_stopping_batch.update_history(validation_loss)
    #if early_stopping_batch.should_stop():
    ## if early_stopping_batch.early_stop:
        #print(f"\nEarly stopping triggered for validation loss = {validation_loss}\n")
        #break

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
    print(f'F1-score per class: {f1_scores}\n')
    viz.plot_lines('Validation Loss', validation_loss)
    viz.plot_lines('Validation Accuracy', accuracy)
    viz.plot_lines('Precision', precision)
    viz.plot_lines('Recall', recall)
    viz.plot_lines('F1-scores', f1_scores)

    return validation_loss, accuracy


"""  **** Grid search loop ****  """

# Define the grid for hyperparameters
max_norms = [8, 4, 2]
batch_sizes = [32, 16]
transformer_layers_options = [1]
num_dense_layers_options = [2, 0]
num_heads_options = [16, 8]
embedding_dimensions = [128]
weight_decays = [0.5e-4, 0.5e-5, 0.5e-6, 0.5e-6]

print(f'\nMax norms for gradients clipping = {max_norms}')
print(f'weight_decays = {weight_decays}')
print(f'batch sizes = {batch_sizes}')
print(f'transformer layers = {transformer_layers_options}')
print(f'num dense layers = {num_dense_layers_options}')
print(f'num heads = {num_heads_options}')
print(f'embedding dimensions = {embedding_dimensions}\n')

best_accuracy = 0  # Track the best accuracy
best_hyperparameters = None  # Track the best hyperparameters

for max_norm in max_norms:
    for weight_decay in weight_decays:

        # Memory-affecting loops
        for batch_size in batch_sizes:

            # Create data loaders
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

            for transformer_layers in transformer_layers_options:
                for num_dense_layers in num_dense_layers_options:
                    for num_heads in num_heads_options:
                        for embedding_dimension in embedding_dimensions:
                            # Clear all windows in visdom display
                            vis.delete_env('Astro Classifier')

                            # Set a seed by hand to avoid unpredictable results
                            torch.manual_seed(3908274565)

                            fc_out_dim = embedding_dimension
                            dense_dims = [fc_out_dim * 4, fc_out_dim * 2, fc_out_dim] # List of output dimensions for dense layers # The best by now
                            cnn_out_dim = 2 * dense_dims[0]
                            cnn_out_dims = [cnn_out_dim // 8, cnn_out_dim // 4, cnn_out_dim // 2, cnn_out_dim] # List of output dimensions for convolutional layers

                            print(f'\nConvolutional layers = {cnn_out_dims}')
                            print(f'Full connected layers = {dense_dims}')
                            print(f'Max norm for gradients clipping = {max_norm}')
                            print(f'weight_decay = {weight_decay}')
                            print(f'Batch size = {batch_size}')
                            print(f'Transformer layers = {transformer_layers}')
                            print(f'Num dense layers = {num_dense_layers}')
                            print(f'Num heads = {num_heads}')
                            print(f'Embedding dimension of the Encoder Attention = {embedding_dimension}\n')

                            # Instantiate the CNN + Dense layer + Transformer
                            model = CNNTransformer(
                                                CNN(cnn_out_dims, dense_dims),
                                                num_heads=num_heads,
                                                transformer_layers=transformer_layers,
                                                num_dense_layers=num_dense_layers
                                                )

                            # Restart all the network weights:
                            model.apply(init_weights)

                            # Move the model to the GPU device
                            model.to(device)

                            # Define the loss function and optimizer
                            criterion = nn.CrossEntropyLoss(weight=class_weights)
                            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=.5e-6)

                            # Re-instantiating different schedulers to adjust the learning rate
                            scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5, verbose=True)
                            # More radical decrease in case of plateau detection
                            scheduler_by_accuracy = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
                            scheduler_by_valloss = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

                            # Re-instantiating the objects of early stopping
                            early_stopping_batch = EarlyStoppingBatch(patience=30)
                            early_stopping_valloss = EarlyStoppingValLoss(patience=20)
                            early_stopping_accuracy = EarlyStoppingAccuracy(patience=20)

                            # This snippet was proposed by Chat GPT-4 to avoid exiting on out-of-memory runtime error
                            try:
                                train_and_validate(model, train_dataloader, validation_dataloader, criterion, optimizer, max_norm)
                            except RuntimeError as e:
                                if 'out of memory' in str(e):
                                    print("WARNING: Out of memory. Skipping grid element")
                                    # Handle the out-of-memory issue here, e.g., by reducing batch size or skipping
                                    continue
                                else:
                                    raise e  # Re-raise the exception if it's not a memory error

                            # Evaluate the model and update best_hyperparameters if it's the best one yet
                            _, current_accuracy = validate(model, validation_dataloader)
                            if current_accuracy > best_accuracy:
                                best_accuracy = current_accuracy
                                best_hyperparameters = (batch_size, transformer_layers, num_dense_layers, num_heads, embedding_dimension)

# Grid loop ends down here

# Print out the best hyperparameter set and its performance
print("\nBest Hyperparameters:")
print(f'Max norm for gradients clipping = {max_norm}')
print(f'weight_decay = {weight_decay}')
print(f"Batch Size={best_hyperparameters[0]}")
print(f"Transformer Layers={best_hyperparameters[1]}")
print(f"Dense Layers={best_hyperparameters[2]}")
print(f"Heads={best_hyperparameters[3]}")
print(f"Embedding dimension={best_hyperparameters[4]}\n")
print(f"\nYielded Best Accuracy: {best_accuracy}")
