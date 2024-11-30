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

def get_mnist_loaders(batch_size=128):
    """Create MNIST train and test data loaders"""
    # Get data directory path
    data_path = get_data_path()
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=5,  # Slight rotation (-5 to +5 degrees)
            translate=(0.05, 0.05),  # Small random shifts
            scale=(0.95, 1.05),  # Slight scaling
            fill=0  # Fill empty areas with black
        ),
        transforms.RandomPerspective(
            distortion_scale=0.2,
            p=0.2,  # Apply perspective transform 20% of the time
            fill=0
        ),
        transforms.ToTensor(),
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