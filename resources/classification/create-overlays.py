import numpy as np
import os
from PIL import Image
from torchvision import datasets, transforms

# Define the transformation for MNIST images
mnist_transform = transforms.Compose([
    transforms.Resize((56, 56)),  # Resize to 84x84
    transforms.ToTensor()
])

# Define the transformation for CIFAR-10 images
cifar_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 84x84
    transforms.ToTensor()
])

# Download MNIST training and test datasets
mnist_train_dataset = datasets.MNIST(root='./data', train=True, transform=mnist_transform, download=True)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, transform=mnist_transform, download=True)

# Download CIFAR-10 training and test datasets
cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, transform=cifar_transform, download=True)
cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, transform=cifar_transform, download=True)

# Load the empty viewport frame
frame = Image.open('empty-viewport.png')

def overlay_image_on_frame(img, frame):
    # Calculate the position to paste the image onto the center of the frame
    x_offset = (frame.width - img.width) // 2
    y_offset = (frame.height - img.height) // 2
    frame.paste(img, (x_offset, y_offset))
    return frame

# Function to process and save the dataset
def process_and_save(dataset, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for idx, (image, label) in enumerate(dataset):
        # Convert tensor back to PIL Image
        image = transforms.ToPILImage()(image)
        overlaid_image = overlay_image_on_frame(image, frame.copy())
        
        # Create a subfolder for each class if it doesn't exist
        class_folder = os.path.join(folder_name, str(label))
        os.makedirs(class_folder, exist_ok=True)
        
        overlaid_image.save(os.path.join(class_folder, f"img_{idx}.png"))

# Process and save the MNIST training and test datasets
process_and_save(mnist_train_dataset, 'mnist_train_overlay')
process_and_save(mnist_test_dataset, 'mnist_test_overlay')

# Process and save the CIFAR-10 training and test datasets
process_and_save(cifar_train_dataset, 'cifar_train_overlay')
process_and_save(cifar_test_dataset, 'cifar_test_overlay')

# Delete the data directory
os.system('rm -rf data')

