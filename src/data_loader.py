import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_dataset(dataset_path, validation_size=1000, batch_size=64, img_size=(256, 256)):
    """
    Loads the FFHQ dataset, preprocesses it, and returns train and validation DataLoaders.

    Args:
        dataset_path (str): Path to the dataset.
        validation_size (int): Number of images to reserve for validation.
        img_size (tuple): Target size for image resizing (width, height).

    Returns:
        tuple: train_loader, val_loader
    """
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")

    print(f"Using device: {device}")

     # Load dataset and resize to target img_size and convert to tensor.
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)


    return 0
