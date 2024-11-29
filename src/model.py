import torch
import torch.nn as nn
import torch.nn.functional as F

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # Input: 1x28x28 -> Output: 8x28x28
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # Input: 4x14x14 -> Output: 8x14x14

        # 1D Convolution for channel reduction
        self.conv1d_1 = nn.Conv1d(8, 4, kernel_size=1)  # Input: (batch, 8, 196) -> Output: (batch, 4, 196)
        self.bn1d_1 = nn.BatchNorm1d(4)  # Match the output channels of conv1d_1

        # Fully connected layer
        self.fc1 = nn.Linear(8 * 7 * 7, 10)  # Input: flattened 8x7x7 -> Output: 10 classes

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)

        # Dropout
        self.dropout = nn.Dropout(0.05)  # Slightly reduced dropout

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)  # Output: (batch, 8, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Max Pooling
        x = F.max_pool2d(x, 2)  # Output: (batch, 8, 14, 14)

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 8, -1)  # Reshape to (batch, channels, height * width) = (batch, 8, 196)

        # 1D Convolution
        x = self.conv1d_1(x)  # Output: (batch, 4, 196)
        x = self.bn1d_1(x)

        # Reshape back to 2D
        x = x.view(batch_size, 4, 14, 14)  # Reshape to (batch, channels, height, width)

        # Second convolutional block
        x = self.conv2(x)  # Output: (batch, 8, 14, 14)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, 2)  # Output: (batch, 8, 7, 7)

        # Flatten and fully connected layer
        x = x.view(-1, 8 * 7 * 7)  # Flatten to (batch, 8 * 7 * 7)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)