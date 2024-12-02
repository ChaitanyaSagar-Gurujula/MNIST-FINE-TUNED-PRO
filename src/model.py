import torch
import torch.nn as nn
import torch.nn.functional as F

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # Input: 1x28x28 -> Output: 4x28x28
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # Input: 4x28x28 -> Output: 8x28x28
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # Input: 4x14x14 -> Output: 8x14x14
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Input: 8x14x14 -> Output: 16x14x14
        self.conv5 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Input: 8x7x7 -> Output: 16x7x7
        self.conv6 = nn.Conv2d(16, 24, kernel_size=3, padding=1)  # Input: 16x7x7 -> Output: 32x7x7


        # Batch normalization
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(24)
        self.bn_fc1 = nn.BatchNorm1d(32)

        # 1D Convolution for channel reduction
        self.conv1d_1 = nn.Conv1d(8, 4, kernel_size=1)  # Input: (batch, 8, 196) -> Output: (batch, 4, 196)
        self.bn1d_1 = nn.BatchNorm1d(4)  # Match the output channels of conv1d_1
        self.conv1d_2 = nn.Conv1d(16, 8, kernel_size=1)  # Input: (batch, 16, 49) -> Output: (batch, 8, 49)
        self.bn1d_2 = nn.BatchNorm1d(8)  # Match the output channels of conv1d_2

        # Fully connected layer
        self.fc1 = nn.Linear(24*2*2, 10)  # Input: flattened 32*2*2 -> Output: 10 classes
        #self.fc2 = nn.Linear(32, 10)  # Input: flattened 32*2*2 -> Output: 10 classes

        # Dropout
        self.dropout_1 = nn.Dropout(0.05)  # Slightly reduced dropout
        # Dropout
        self.dropout_2 = nn.Dropout(0.01)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)  # Output: (batch, 4, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.conv2(x)  # Output: (batch, 8, 28, 28)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = F.max_pool2d(x, 2)  # Output: (batch, 8, 14, 14)

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 8, -1)  # Reshape to (batch, channels, height * width) = (batch, 8, 196)

        # 1D Convolution
        x = self.conv1d_1(x)  # Output: (batch, 4, 196)
        x = self.bn1d_1(x)
        x = F.relu(x)

        # Reshape back to 2D
        x = x.view(batch_size, 4, 14, 14)  # Reshape to (batch, channels, height, width)

        # Second convolutional block
        x = self.conv3(x)  # Output: (batch, 8, 14, 14)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.conv4(x)  # Output: (batch, 16, 14, 14)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = F.max_pool2d(x, 2)  # Output: (batch, 16, 7, 7)

        # Reshape for 1D convolution
        x = x.view(batch_size, 16, -1)  # Reshape to (batch, channels, height * width) = (batch, 16, 49)

        # 1D Convolution
        x = self.conv1d_2(x)  # Output: (batch, 8, 49)
        x = self.bn1d_2(x)
        x = F.relu(x)

        # Reshape back to 2D
        x = x.view(batch_size, 8, 7, 7)  # Reshape to (batch, channels, height, width)

        # Third convolutional block
        x = self.conv5(x)  # Output: (batch, 16, 7, 7)
        x = self.bn5(x)
        x = F.relu(x)
        #x = self.dropout_2(x)
        x = self.conv6(x)  # Output: (batch, 32, 7, 7)
        x = self.bn6(x)
        x = F.relu(x)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (2, 2))  # Output: (batch, 32, 4, 4)
        #x = F.adaptive_avg_pool2d(x, (2, 2))  # Output: (batch, 32, 2, 2)

        # Flatten and fully connected layer
        x = x.view(batch_size, -1)  # Flatten to (batch, 32*2*2)
        x = self.fc1(x)
        #x = self.bn_fc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)

        return F.log_softmax(x, dim=1)