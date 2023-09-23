import numpy as np
import os
from PIL import Image
from torchvision import datasets, transforms

# Define the transformation for MNIST images
transform = transforms.Compose([
    transforms.Resize((56, 56)),  # Resize to 84x84
    transforms.ToTensor()
])

# Download MNIST training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Load the empty viewport frame
frame = Image.open('empty-viewport.png')

def overlay_mnist_on_frame(mnist_img, frame):
    # Calculate the position to paste the MNIST image onto the center of the frame
    x_offset = (frame.width - mnist_img.width) // 2
    y_offset = (frame.height - mnist_img.height) // 2
    # Convert grayscale MNIST image to have an alpha channel
    mnist_img_alpha = mnist_img.convert("L").convert("RGBA")
    # Set alpha channel to fully opaque
    mnist_img_alpha.putalpha(255)
    frame.paste(mnist_img_alpha, (x_offset, y_offset), mnist_img_alpha)
    return frame

# Function to process and save the dataset
def process_and_save(dataset, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for idx, (image, label) in enumerate(dataset):
        # Convert tensor back to PIL Image
        image = transforms.ToPILImage()(image)
        # Convert grayscale to RGB
        image = image.convert("RGB")
        overlaid_image = overlay_mnist_on_frame(image, frame.copy())
        
        # Create a subfolder for each class if it doesn't exist
        class_folder = os.path.join(folder_name, str(label))
        os.makedirs(class_folder, exist_ok=True)
        
        overlaid_image.save(os.path.join(class_folder, f"img_{idx}.png"))


# Process and save the training and test datasets
process_and_save(train_dataset, 'mnist_train_overlay')
process_and_save(test_dataset, 'mnist_test_overlay')

# Delete the data directory
os.system('rm -rf data')


