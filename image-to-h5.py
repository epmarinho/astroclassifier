# Authors: Eraldo Pereira Marinho and ChatGPT
# Sep 11, 2023, 4:23pm

import h5py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Define a function to convert a PyTorch DataLoader to H5 format
def dataloader_to_h5(loader, h5file, dataset_name, class_labels):
    data_list = []
    labels_list = []

    for inputs, labels in loader:
        data_list.append(inputs.numpy())  # Convert PyTorch tensor to NumPy array
        labels_list.append(labels.numpy())  # Convert PyTorch tensor to NumPy array

    data_array = np.concatenate(data_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    h5file.create_dataset(f"{dataset_name}_data", data=data_array)
    h5file.create_dataset(f"{dataset_name}_labels", data=labels_array)

    # Store class labels as an attribute
    h5file.attrs[f"{dataset_name}_class_labels"] = class_labels

nmaxpool = 3
img_width = 256
img_height = 256
img_out_width = img_width // 2**nmaxpool
img_out_height = img_height // 2**nmaxpool

# Transformations for preprocessing
transform_train = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((img_width, img_height)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly adjusts brightness, contrast, saturation, and hue.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_validation = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define the root directory of your dataset
train_data_root = r'images/train'
validation_data_root = r'images/validation'

# Create ImageFolder datasets to infer class labels
train_dataset = ImageFolder(root=train_data_root, transform=transform_train)
validation_dataset = ImageFolder(root=validation_data_root, transform=transform_validation)

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

# Create data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Create an H5 file to store the data
h5file_path = "datasets.h5"
with h5py.File(h5file_path, "w") as h5file:
    dataloader_to_h5(train_loader, h5file, "train", class_labels)
    dataloader_to_h5(validation_loader, h5file, "validation", class_labels)

# Now, your data from the data loaders is stored in "datasets.h5" in H5 format with inferred class labels.
