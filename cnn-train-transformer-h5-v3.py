""" Code name: cnn-train-transformer-h5-v3.py (main code for training)"""

# Author: Eraldo Pereira Marinho, Ph.D
# About: The code imports cnn_transformer_core to allow Transformer+CNN to classify astronomical images
# Creation: Jul 12, 2023

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
# import torchvision.transforms as transforms
import visdom
from utils import Visualizer
import os
import torch.nn.init as init
import numpy as np
# from PIL import Image
import pillow_avif

# Set a seed by hand to avoid unpredictable results
torch.manual_seed(3908274)

# Trying to minimize the randomness problem - maybe not enough
torch.backends.cudnn.deterministic = True

from cnn_transformer_core_h5_v3 import model
from cnn_transformer_core_h5_v3 import train_dataloader
from cnn_transformer_core_h5_v3 import validation_dataloader
# from cnn_transformer_core_h5_v3 import learning_rate
# from cnn_transformer_core_h5_v3 import num_epochs
# from cnn_transformer_core_h5_v3 import Swish

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {device}")

viz = Visualizer.Visualizer('Astro Classifier', use_incoming_socket=False)

# Define the class weight vector empirically obtained from the last run:
# run after the classes histogram:
galaxies = np.float32(1/226)
globular = np.float32(1/109)
nebulae  = np.float32(1/200)
openclust= np.float32(1/117)
## run this before to have an actual class histogram (should be?)
#galaxies = np.float32(1)
#globular = np.float32(1)
#nebulae  = np.float32(1)
#openclust= np.float32(1)
norm_denominator=galaxies + globular + nebulae + openclust
weight_class_0=galaxies/norm_denominator
weight_class_1=globular/norm_denominator
weight_class_2=nebulae/norm_denominator
weight_class_3=openclust/norm_denominator
# Instantiate the class weight tensor
class_weights = torch.tensor([weight_class_0, weight_class_1, weight_class_2, weight_class_3])
print(f"\nClass weights = {class_weights}\n")
# Weights tensor must be converted to the set device
class_weights = class_weights.to(device)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Initialize weights using Xavier uniform initialization
        init.xavier_uniform_(m.weight)

        # Set biases to zero if they exist
        if m.bias is not None:
            init.constant_(m.bias, 0)

# Training parameters

num_epochs = 100

initial_learning_rate = 1e-4

# Define the loss function and optimizer
loss_func = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=5e-7)

# Define a scheduler to adjust the learning rate for each peculiarity
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=False)
scheduler_by_accuracy = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
scheduler_by_valloss = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Check if the pretrained file exists
model_checkpoint = "trained_cnn_model.pth"
if os.path.exists(model_checkpoint):
    # Load pretrained weights
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint)
    print(f"\nPretrained weights \"{model_checkpoint}\" found.\n")
else:
    print("\nNo pretrained weights file found. Initializing with PyTorch default weights.\n")
#     # He initialization in PyTorch
#     # Access all the linear layers (fully connected)
#     # Ensure the model contains only layers that should be initialized with He
    # for layer in model.children():
    #     if isinstance(layer, nn.Linear):
    #         init.kaiming_normal_(layer.weight)

# # Check the loaded weights
# print(model.state_dict())

# This is basically my earling stopping proposed in previously unpublished works
#class EarlyStoppingBatch:
    #def __init__(self, remembrance, patience,  threshold=.005):
        #self.remembrance = remembrance
        #self.history = []
        #self.patience = patience
        ##self.patience = patience if patience <= remembrance else remembrance
        #self.threshold = threshold

    #def update_history(self, new_loss):
        ## Update the history array with the new loss value
        #self.history.append(new_loss)
        ## Keep only the most recent 'remembrance' elements
        #if len(self.history) > self.remembrance:
            #self.history.pop(0) # Discard the earliest one

    #def should_stop(self):
        ## Check if the minimum loss in the history is repeated or becomes smaller
        ## if len(self.history) < self.remembrance:
        #if len(self.history) < self.patience and self.remembrance >= self.patience:
            #return False  # Not enough data to decide
        #if len(self.history) < self.remembrance and self.remembrance < self.patience:
            #return False
        #return self.history[-1] <= min(self.history[:-1]) + self.threshold

#early_stopping_batch = EarlyStoppingBatch(remembrance=20, patience=30)

#class EarlyStoppingValLoss:
    #def __init__(self, remembrance, patience,  threshold=.005):
        #self.remembrance = remembrance
        #self.history = []
        #self.patience = patience
        ##self.patience = patience if patience <= remembrance else remembrance
        #self.threshold = threshold

    #def update_history(self, new_loss):
        ## Update the history array with the new loss value
        #self.history.append(new_loss)
        ## Keep only the most recent 'remembrance' elements
        #if len(self.history) > self.remembrance:
            #self.history.pop(0) # Discard the earliest one

    ##def should_stop(self):
        ### Check if the minimum loss in the history is repeated or becomes smaller
        ### if len(self.history) < self.remembrance:
        ##if len(self.history) < self.patience:
            ##return False  # Not enough data to decide
        ##return min(self.history[:-1]) - self.threshold <= self.history[-1] <= min(self.history[:-1]) + self.threshold

    #def should_stop(self):
        ## Check if we have enough data to make a decision
        #if len(self.history) < self.patience and self.remembrance >= self.patience:
            #return False  # Not enough data to decide
        #if len(self.history) < self.remembrance and self.remembrance < self.patience:
            #return False

        ## Check the best loss so far
        #max_accuracy = min(self.history[:-1])

        ## Check if the loss has not improved significantly for 'patience' epochs
        #plateau_count = sum(1 for x in self.history[-self.patience:] if max_accuracy - self.threshold <= x <= max_accuracy + self.threshold)

        ## If the loss has been on a plateau for 'patience' consecutive epochs, stop
        #return plateau_count >= self.patience

#early_stopping_valloss = EarlyStoppingValLoss(remembrance=20, patience=25)

#class EarlyStoppingAccuracy:
    #def __init__(self, remembrance, patience,  threshold=.0005):
        #self.remembrance = remembrance
        #self.history = []
        #self.patience = patience
        ##self.patience = patience if patience <= remembrance else remembrance
        #self.threshold = threshold

    #def update_history(self, new_accuracy):
        ## Update the history array with the new loss value
        #self.history.append(new_accuracy)
        ## Keep only the most recent 'remembrance' elements
        #if len(self.history) > self.remembrance:
            #self.history.pop(0) # Discard the earliest one

    #def should_stop(self):
        ## Check if the minimum loss in the history is repeated or becomes smaller
        ## if len(self.history) < self.remembrance:
        #if len(self.history) < self.patience and self.remembrance >= self.patience:
            #return False  # Not enough data to decide
        #if len(self.history) < self.remembrance and self.remembrance < self.patience:
            #return False
        #return self.history[-1] >= max(self.history[:-1]) - self.threshold

#early_stopping_accuracy = EarlyStoppingAccuracy(remembrance=20, patience=20)

class EarlyStoppingBatch:
    def __init__(self, patience=30, threshold=.005):
        """
        Initialize the EarlyStoppingBatch object.

        Parameters:
        - patience: int, number of epochs with no improvement after which training will be stopped.
        - threshold: float, the minimum change in the monitored quantity to qualify as an improvement.
        """
        self.history = []  # List to store the history of loss values
        self.patience = patience  # How many epochs to wait before stopping
        self.threshold = threshold  # The threshold for determining if the model has improved

    def update_history(self, new_loss):
        """
        Update the history list with the new loss value.

        Parameters:
        - new_loss: float, the loss value from the current epoch.
        """
        self.history.append(new_loss)  # Add the new loss to the history
        if len(self.history) > self.patience:
            self.history.pop(0)  # Remove the oldest loss if history exceeds patience

    def should_stop(self):
        """
        Determine if training should be stopped.

        Returns:
        - Boolean, True if training should be stopped, False otherwise.
        """
        if len(self.history) < self.patience:
            return False  # Not enough data to decide, continue training
        # Stop if the most recent loss is not significantly lower than the best previous loss
        return self.history[-1] <= min(self.history[:-1]) + self.threshold

class EarlyStoppingValLoss:
    def __init__(self, patience=30, threshold=.005):
        """
        Initialize the EarlyStoppingValLoss object.

        Parameters:
        - patience: int, number of epochs with no improvement after which training will be stopped.
        - threshold: float, the minimum change in the monitored quantity to qualify as an improvement.
        """
        self.history = []  # List to store the history of validation loss values
        self.patience = patience  # How many epochs to wait before stopping
        self.threshold = threshold  # The threshold for determining if the model has improved

    def update_history(self, new_loss):
        """
        Update the history list with the new validation loss value.

        Parameters:
        - new_loss: float, the validation loss value from the current epoch.
        """
        self.history.append(new_loss)  # Add the new validation loss to the history
        if len(self.history) > self.patience:
            self.history.pop(0)  # Remove the oldest loss if history exceeds patience

    def should_stop(self):
        """
        Determine if training should be stopped based on validation loss.

        Returns:
        - Boolean, True if training should be stopped, False otherwise.
        """
        if len(self.history) < self.patience:
            return False  # Not enough data to decide, continue training

        # Check the best loss so far
        max_accuracy = min(self.history[:-1])

        # Count how many recent losses are within the threshold of the best loss
        plateau_count = sum(1 for x in self.history[-self.patience:] if max_accuracy - self.threshold <= x <= max_accuracy + self.threshold)

        # Stop if the loss hasn't improved for 'patience' consecutive epochs
        return plateau_count >= self.patience



class EarlyStoppingAccuracy:
    def __init__(self, patience=30, threshold=.005):
        """
        Initialize the EarlyStoppingAccuracy object.

        Parameters:
        - patience: int, number of epochs with no improvement after which training will be stopped.
        - threshold: float, the minimum change in the monitored quantity to qualify as an improvement.
        """
        self.history = []  # List to store the history of accuracy values
        self.patience = patience  # How many epochs to wait before stopping
        self.threshold = threshold  # The threshold for determining if the model has improved

    def update_history(self, new_accuracy):
        """
        Update the history list with the new accuracy value.

        Parameters:
        - new_accuracy: float, the accuracy value from the current epoch.
        """
        self.history.append(new_accuracy)  # Add the new accuracy to the history
        if len(self.history) > self.patience:
            self.history.pop(0)  # Remove the oldest accuracy if history exceeds patience

    #def should_stop(self):
        #"""
        #Determine if training should be stopped based on accuracy.

        #Returns:
        #- Boolean, True if training should be stopped, False otherwise.
        #"""
        #if len(self.history) < self.patience:
            #return False  # Not enough data to decide, continue training
        ## Stop if the most recent accuracy is not significantly higher than the best previous accuracy
        #return self.history[-1] >= max(self.history[:-1]) - self.threshold

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

early_stopping_batch = EarlyStoppingBatch(patience=20)
early_stopping_valloss = EarlyStoppingValLoss(patience=20)
early_stopping_accuracy = EarlyStoppingAccuracy(patience=20)

# Restart all the network weights:
model.apply(init_weights)

# Move the model to the GPU device
model.to(device)

# Training function
def train_and_validate(model, dataloader, validation_loader, loss_func, optimizer):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            # Move the images and labels to the GPU device
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Pass the input images through the CNN+Transformer
            outputs = model(images)

            # Get the output of the loss function
            loss = loss_func(outputs, labels)

            # Compute gradients to be used by gradient descent in optimizer step
            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)

            # Optimization step
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

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
        viz.plot_lines('Batch Loss', epoch_loss)

        early_stopping_batch.update_history(epoch_loss)
        if early_stopping_batch.should_stop():
            print(f"\nEarly stopping triggered at epoch {epoch + 1} for batch loss = {epoch_loss}\n")
            break

        validation_loss, accuracy = validate(model, validation_loader)

        early_stopping_valloss.update_history(validation_loss)
        if early_stopping_valloss.should_stop():
            print(f"\nEarly stopping triggered at epoch {epoch + 1} for validation loss = {validation_loss}\n")
            break

        early_stopping_accuracy.update_history(accuracy)
        if early_stopping_accuracy.should_stop():
            print(f"\nEarly stopping triggered at epoch {epoch + 1} for validation accuracy = {accuracy:.2f}%\n")
            break

        # Update the learning rate based on the scheduler
        scheduler_by_accuracy.step(accuracy)
        scheduler_by_valloss.step(validation_loss)

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

            # Compute the Loss function defined in loss_func
            loss = loss_func(outputs, true_labels)

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
    specificity = []

    for i in range(len(cm)):
        true_positive = cm[i, i]
        false_positive = sum(cm[j, i] for j in range(len(cm))) - true_positive
        false_negative = sum(cm[i, j] for j in range(len(cm))) - true_positive
        true_negative = sum(cm[j, k] for j in range(len(cm)) for k in range(len(cm))) - (true_positive + false_positive + false_negative)

        precision_i = true_positive / (true_positive + false_positive + 1e-12)
        recall_i = true_positive / (true_positive + false_negative + 1e-12)
        f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i + 1e-12)
        specificity_i = true_negative / (true_negative + false_positive + 1e-12)

        precision.append(precision_i)
        recall.append(recall_i)
        f1_scores.append(f1_i)
        specificity.append(specificity_i)

    print(f'Validation Loss: {validation_loss:.6f}, Validation Accuracy: {accuracy:.2f}%')
    print(f'Precision per class: {precision}')
    print(f'Recall per class: {recall}')
    print(f'F1-score per class: {f1_scores}')
    print(f'Specificity per class: {specificity}')
    viz.plot_lines('Validation Loss', validation_loss)
    viz.plot_lines('Validation Accuracy', accuracy)
    viz.plot_lines('Precision', precision)
    viz.plot_lines('Recall', recall)
    viz.plot_lines('F1-scores', f1_scores)
    viz.plot_lines('Specificity-scores', specificity)

    return validation_loss, accuracy

# Train the CNN+Transformer
train_and_validate(model, train_dataloader, validation_dataloader, loss_func, optimizer)

# Validate the CNN+Transformer
validate(model, validation_dataloader)

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

label_count = torch.bincount(torch.tensor(indices, dtype=torch.int64))

print("Label count: ", label_count)

# plt.bar(torch.arange(len(label_count)), label_count)
# plt.xlabel('Labels')
# plt.ylabel('Frequency')
# plt.show()
