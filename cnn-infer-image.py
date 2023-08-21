import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from utils import Visualizer  # You should import your Visualizer module here

# hyperparameters
import hyperparms
nmaxpool = hyperparms.nmaxpool
img_width = hyperparms.img_width
img_height = hyperparms.img_height
img_out_width = hyperparms.img_out_width
img_out_height = hyperparms.img_out_height
num_classes = hyperparms.num_classes

# Create an instance of the CNN model
loaded_model = hyperparms.CNN(num_classes=num_classes)

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

# Load a sample image for inference
sample_image_path = 'images/pexels-photo-816608.jpeg'  # Replace with the actual path
print(sample_image_path)
sample_image = Image.open(sample_image_path).convert("RGB")
input_image = transform(sample_image).unsqueeze(0)  # Add an extra dimension for the batch

# Make sure to move the input image to the same device as the model (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_image = input_image.to(device)

# Perform inference
with torch.no_grad():
    output = loaded_model(input_image)

# Get the predicted class index
predicted_class_index = torch.argmax(output).item()

# Replace 'class_labels' with your actual class labels
class_labels = ['galaxies', 'globular clusters', 'nebulae', 'open clusters', 'others']

# Print the inferred class
print(f"Inferred class: {class_labels[predicted_class_index]}")
