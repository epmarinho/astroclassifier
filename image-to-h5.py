""" Code image-to-h5.py: converts an image directory structure into an H5 dataset"""

# Authors: Eraldo Pereira Marinho and ChatGPT
# Sep 11, 2023, 4:23pm

import h5py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, features
import pillow_avif
from torchvision.datasets import ImageFolder
# from cnn_transformer_core_h5_v3 import batch_size

batch_size = 32

# Define a function to convert a PyTorch DataLoader to H5 format
def dataloader_to_h5(loader, h5file, dataset_name, class_labels):
    data_list = []
    labels_list = []

    for inputs, labels in loader:
        try:
            data_list.append(inputs.numpy())  # Convert PyTorch tensor to NumPy array
            labels_list.append(labels.numpy())  # Convert PyTorch tensor to NumPy array
        except Exception as e:
            print(f"Error loading data from DataLoader: {e}")
            continue

    if len(data_list) == 0 or len(labels_list) == 0:
        print("No valid data to store in H5 file.")
        return

    data_array = np.concatenate(data_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    try:
        h5file.create_dataset(f"{dataset_name}_data", data=data_array)
        h5file.create_dataset(f"{dataset_name}_labels", data=labels_array)
    except Exception as e:
        print(f"Error creating datasets in H5 file: {e}")

    # Store class labels as an attribute
    try:
        h5file.attrs[f"{dataset_name}_class_labels"] = class_labels
    except Exception as e:
        print(f"Error storing class labels as attributes in H5 file: {e}")

# Image dimensions
image_size = (256,256)
#crop_size = (224,224)

# Padd input images to the minimal square frame
def pad_to_square(img):
    # Compute the difference between the longest and shortest side
    w, h = img.size
    diff = abs(h - w) // 2

    # Determine padding for height and width
    pad_h = diff if h <= w else 0
    pad_w = diff if w < h else 0

    # Return a new padded PIL image
    return transforms.functional.pad(img, (pad_w, pad_h, pad_w, pad_h))

# # Compute the dataset-wide mean and standard deviation
# def compute_mean_std(loader):
#     channel_sum, channel_squared_sum, n_batches = 0, 0, 0
#
#     for data, _ in loader:
#         channel_sum += torch.mean(data, dim=[0,2,3])
#         channel_squared_sum += torch.mean(data ** 2, dim=[0,2,3])
#         n_batches += 1
#
#     mean = channel_sum / n_batches
#     std = (channel_squared_sum/n_batches - mean**2)**0.5
#
#     return mean, std

# Define the root directory of your dataset
train_data_root = r'images/train'
validation_data_root = r'images/validation'

# # Create a temporary DataLoader for computing mean and stddev for training data
# temp_train_loader = torch.utils.data.DataLoader(
#     ImageFolder(root=train_data_root, transform=transforms.Compose([
#         transforms.Lambda(pad_to_square),
#         transforms.Resize((image_size, img_height)),
#         transforms.RandomCrop(crop_size),
#         transforms.ToTensor()
#     ])),
#     batch_size=batch_size, shuffle=False
# )
#
# train_mean, train_stddev = compute_mean_std(temp_train_loader)
#
# # Create a temporary DataLoader for computing mean and stddev for training data
# temp_validation_loader = torch.utils.data.DataLoader(
#     ImageFolder(root=validation_data_root, transform=transforms.Compose([
#         transforms.Lambda(pad_to_square),
#         transforms.Resize((image_size, img_height)),
#         transforms.RandomCrop(crop_size),
#         transforms.ToTensor()
#     ])),
#     batch_size=batch_size, shuffle=False
# )
#
# validation_mean, validation_stddev = compute_mean_std(temp_validation_loader)

# Transformations for preprocessing

transform_train = transforms.Compose([
    transforms.Lambda(pad_to_square), # Apply padding to maintain aspect ratio as suggested by GPT-4
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally (left to right)
    transforms.RandomAdjustSharpness(sharpness_factor=4),
    transforms.Resize(image_size),
    #transforms.RandomCrop(crop_size),
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # transformes color ranges from [0,1] to [-1,1]
    # transforms.Normalize(mean=train_mean, std=train_stddev) # This did not work
])

transform_validation = transforms.Compose([
    transforms.Lambda(pad_to_square), # Apply padding to maintain aspect ratio # Suggested by GPT-4
    transforms.Resize(image_size),
    #transforms.RandomCrop(crop_size),
    transforms.ToTensor(), # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # transformes color ranges from [0,1] to [-1,1]
])

# Create ImageFolder datasets to infer class labels
train_dataset = ImageFolder(root=train_data_root, transform=transform_train)
validation_dataset = ImageFolder(root=validation_data_root, transform=transform_validation)

## Only use the following to check the default tensor normalization
# is_normalized = True  # Assume the dataset is normalized

# for image, _ in train_dataset:
#     min_pixel_value = torch.min(image)
#     max_pixel_value = torch.max(image)

    # if min_pixel_value != 0.0 or max_pixel_value != 1.0:
    #     is_normalized = False
    #     break

# if is_normalized:
#     print("The images are normalized to [0, 1].")
# else:
#     print("The images are not normalized to [0, 1].")
# print(f" Min pix = {min_pixel_value}, max pix = {max_pixel_value}.")

# Get the class labels from the dataset
class_labels = train_dataset.classes

# Check images integrity for training/validation images
from tqdm import tqdm # tqdm allows displaying a progress bar
import os
def check_images(image_folder):
    for subdir in os.listdir(image_folder):
        subdir_path = os.path.join(image_folder, subdir)

        # Check if the item in validation_data_root is a directory
        if os.path.isdir(subdir_path):
            for filename in tqdm(os.listdir(subdir_path), desc=f"Processing Images in {subdir}"):
                image_path = os.path.join(subdir_path, filename)

                try:
                    sample_image = Image.open(image_path).convert("RGB")
                    # Continue with preprocessing and inference here
                except Exception as e:
                    print(f"Error loading image {image_path}: {str(e)}")

# Create data loaders

print("\n### Checking training images\n")
check_images(train_data_root)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("\n### Checking validation images\n")
check_images(validation_data_root)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# Create an H5 file to store the data

print("\n### Converting images data-loader to H5 format\n")
h5file_path = "datasets.h5"
with h5py.File(h5file_path, "w") as h5file:
    dataloader_to_h5(train_loader, h5file, "train", class_labels)
    dataloader_to_h5(validation_loader, h5file, "validation", class_labels)
print("\n### Done\n")
