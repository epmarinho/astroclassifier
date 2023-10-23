import torch
from torchvision import transforms
from PIL import Image
import pillow_avif
import matplotlib.pyplot as plt
import os

image_dir = 'images/tests'
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.avif', '.webp'))]

image_path = 'images/tests/quasaroutflow.png'
image = Image.open(image_path)

# Image dimensions
image_size = (256,256)
crop_size = (224,224)

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

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Lambda(pad_to_square), # Apply padding to maintain aspect ratio as suggested by GPT-4
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally (left to right)
        transforms.RandomAdjustSharpness(sharpness_factor=4),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), # Randomly adjusts brightness, contrast, saturation, and hue.
        transforms.Resize(image_size),
        transforms.RandomCrop(crop_size),
        # transforms.ToTensor(), # Convert the image to a PyTorch tensor
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # transformes color ranges from [0,1] to [-1,1]
        # transforms.Normalize(mean=train_mean, std=train_stddev) # This did not work
    ])

    transformed_image = transform(image)

    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Transformed Image')
    plt.imshow(transformed_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
