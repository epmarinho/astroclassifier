import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from utils import Visualizer  # You should import your Visualizer module here

# Parameters
num_classes = 4
img_width = 512
img_height = 512
nmaxpool = 6
img_out_width = img_width // (2 ** nmaxpool)
img_out_height = img_height // (2 ** nmaxpool)


# Definir a arquitetura da CNN
class CNN(nn.Module):
    def __init__(self, num_classes=num_classes):
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
        expected_flattened_size = 512 * img_out_width * img_out_height
        self.classifier = nn.Sequential(
            nn.Linear(expected_flattened_size, 1024),
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

# Create an instance of the CNN model
loaded_model = CNN(num_classes=num_classes)

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
sample_image_path = 'images/opo1038b.jpg'  # Replace with the actual path
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
