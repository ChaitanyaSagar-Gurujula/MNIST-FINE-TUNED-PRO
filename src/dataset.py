import os
from pathlib import Path
import torch
from torchvision import datasets, transforms

def get_data_path():
    """Get the path to data directory"""
    # Get the project root directory (assuming we're in src/)
    root_dir = Path(__file__).parent.parent
    # Create data directory if it doesn't exist
    data_dir = root_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)

def get_mnist_loaders(batch_size=128, is_training=True):
    """Create MNIST train and test data loaders"""
    # Get data directory path
    data_path = get_data_path()
    
    if is_training:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
        # 1. Spatial transforms (work on PIL images)
        transforms.RandomAffine(
            degrees=5,
            #translate=(0.05, 0.05),
            shear=(-5, 5),
            fill=0
        ),
        transforms.RandomPerspective(
            distortion_scale=0.2,
            p=0.1,
            fill=0
        ),
        # 2. Convert to tensor
        transforms.ToTensor(),
        # 3. Erasing (works on tensor)
        transforms.RandomErasing(
            p=0.1,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value=0
        ),
        # 4. Normalize (always last)
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        # Training transforms without augmentation
        train_transform = transforms.Compose([
        # 1. Convert to tensor
        transforms.ToTensor(),
        # 2. Normalize (always last)
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Test transforms (no augmentation needed)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets with respective transforms
    train_dataset = datasets.MNIST(
        data_path, train=True, download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        data_path, train=False,
        transform=test_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader 