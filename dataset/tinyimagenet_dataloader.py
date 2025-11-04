import torch.utils.data as data
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
import sys

# --- Configuration ---
# Set the batch size and number of worker processes
BATCH_SIZE = 32
NUM_WORKERS = 8
DATA_ROOT = 'tiny-imagenet/tiny-imagenet-200' # Assumes dataset is extracted here

# --- Data Transformations ---
# Standard transformations required for Tiny-ImageNet
transform = T.Compose([
    T.Resize((224, 224)),  # Resize images to 224x224
    T.ToTensor(),
    # Standard ImageNet normalization values
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

def get_tiny_imagenet_loaders():
    """
    Initializes and returns the DataLoader instances for Tiny-ImageNet.
    
    Checks for the dataset files and handles common exceptions.
    Returns (train_loader, val_loader).
    """
    
    train_dir = os.path.join(DATA_ROOT, 'train')
    val_dir = os.path.join(DATA_ROOT, 'val')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print("🚨 ERROR: Dataset directories not found.")
        print(f"Expected to find: {train_dir} and {val_dir}")
        print("Please run the download/unzip and the reorganization script (utils/data_prep.py) first.")
        return None, None

    try:
        # Load the training and validation datasets using ImageFolder
        tiny_imagenet_dataset_train = ImageFolder(root=train_dir, transform=transform)
        tiny_imagenet_dataset_val = ImageFolder(root=val_dir, transform=transform)
        
        print(f"Found {len(tiny_imagenet_dataset_train)} training images.")
        print(f"Found {len(tiny_imagenet_dataset_val)} validation images.")

        # Create DataLoaders
        train_loader = data.DataLoader(
            tiny_imagenet_dataset_train, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS,
            pin_memory=True # Optimization for CUDA
        )
        val_loader = data.DataLoader(
            tiny_imagenet_dataset_val, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        return train_loader, val_loader

    except Exception as e:
        print(f"An unexpected error occurred during DataLoader creation: {e}")
        return None, None

# Assign loaders for easy import in train.py
train_loader, val_loader = get_tiny_imagenet_loaders()