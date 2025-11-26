import os
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import Subset
import numpy as np

def get_tiny_imagenet_loaders(cfg):
    """
    Create and return train_loader and val_loader using parameters from cfg.
    Nothing is loaded at import time. Everything is built on demand.
    """

    DATA_ROOT = cfg["data_root"]
    BATCH_SIZE = cfg["batch_size"]
    NUM_WORKERS = cfg["num_workers"]
    SUB_RATIO = cfg.get("subset_ratio", 1.0)

    # --- Transforms ---
    transform = T.Compose([
        T.Resize((cfg.get("image_size", 224), cfg.get("image_size", 224))),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(DATA_ROOT, 'train')
    val_dir   = os.path.join(DATA_ROOT, 'val')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Dataset not found in {DATA_ROOT}. Run the dataset prep step first."
        )

    # --- Dataset ---
    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset   = ImageFolder(val_dir, transform=transform)

    print(f"Found {len(train_dataset)} training images.")
    print(f"Found {len(val_dataset)} validation images.")
    
    # --- OPTIONAL: USE ONLY A SUBSET OF THE TRAINING SET FOR FASTER TESTING ---
    
    if SUB_RATIO < 1.0:
        n = len(train_dataset)
        k = int(n * SUB_RATIO)
        indices = np.random.choice(n, k, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"[DATA] Using subset: {k}/{n} ({SUB_RATIO*100:.1f}%)")

    # --- Loaders ---
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
