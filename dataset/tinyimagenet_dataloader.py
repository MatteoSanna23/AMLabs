import torch.utils.data as data
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
import yaml

# --- Load configuration ---
try:
    # Try to load local override first (Colab)
    config_path = "config_local.yaml"
    config = yaml.safe_load(open(config_path))
except FileNotFoundError:
    config_path = "config.yaml"
    config = yaml.safe_load(open(config_path))

# Extract data loader parameters
BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]
DATA_ROOT = config["data_root"]

# --- Data Transformations ---
transform = T.Compose([
    T.Resize((config.get("image_size", 224), config.get("image_size", 224))),  # Resize using config
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_tiny_imagenet_loaders():
    """
    Initializes and returns the DataLoader instances for Tiny-ImageNet.
    Returns (train_loader, val_loader).
    """
    train_dir = os.path.join(DATA_ROOT, 'train')
    val_dir = os.path.join(DATA_ROOT, 'val')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print("🚨 ERROR: Dataset directories not found.")
        print(f"Expected: {train_dir} and {val_dir}")
        print("Please run the download/unzip and reorganization script first.")
        return None, None

    try:
        train_dataset = ImageFolder(root=train_dir, transform=transform)
        val_dataset = ImageFolder(root=val_dir, transform=transform)

        print(f"Found {len(train_dataset)} training images.")
        print(f"Found {len(val_dataset)} validation images.")

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        val_loader = data.DataLoader(
            val_dataset,
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
