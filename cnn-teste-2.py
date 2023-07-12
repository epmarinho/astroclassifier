import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the image size
image_height = 64
image_width = 64
num_channels = 3  # Number of image channels (e.g., 3 for RGB)

# Create a random tensor as the test image
test_image = torch.randn(num_channels, image_height, image_width)

# Convert the tensor to a numpy array
test_image_numpy = test_image.numpy()

# Transpose the numpy array to match the expected shape for displaying an image (H, W, C)
test_image_numpy = test_image_numpy.transpose(1, 2, 0)

# Display the test image using matplotlib
plt.imshow(test_image_numpy)
plt.axis('off')  # Remove the axis labels
plt.show()
