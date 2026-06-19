"""
Data loading utilities for Smart Agriculture project.
Supports PlantVillage and PlantDoc datasets with augmentation.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# --- Normalization stats (ImageNet) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Training augmentations ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
    ),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- Validation / Test transforms ---
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# --- GAN transforms (no normalization, scale to [-1,1]) ---
gan_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def get_dataloaders(train_dir, val_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Returns train and val DataLoaders from ImageFolder directories.

    Args:
        train_dir: Path to training data (ImageFolder structure).
        val_dir: Path to validation data (ImageFolder structure).
        batch_size: Batch size for both loaders.
        num_workers: Number of data loading workers.

    Returns:
        train_loader, val_loader, num_classes
    """
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=test_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_ds.classes)


def get_dataloaders_split(data_dir, val_ratio=0.2, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Loads a single ImageFolder directory and splits into train/val.

    Args:
        data_dir: Path to dataset root (ImageFolder structure).
        val_ratio: Fraction of data to use for validation.
        batch_size: Batch size.
        num_workers: Number of data loading workers.

    Returns:
        train_loader, val_loader, num_classes
    """
    full_ds = datasets.ImageFolder(data_dir, transform=train_transform)
    num_classes = len(full_ds.classes)

    val_size = int(len(full_ds) * val_ratio)
    train_size = len(full_ds) - val_size

    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # Override transform for val split
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=test_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes


def get_gan_dataloader(data_dir, batch_size=64, num_workers=NUM_WORKERS):
    """
    Returns a DataLoader for GAN training (64x64, normalized to [-1,1]).

    Args:
        data_dir: Path to dataset root (ImageFolder structure).
        batch_size: Batch size for GAN training.
        num_workers: Number of data loading workers.

    Returns:
        dataloader, num_classes
    """
    dataset = datasets.ImageFolder(data_dir, transform=gan_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader, len(dataset.classes)
