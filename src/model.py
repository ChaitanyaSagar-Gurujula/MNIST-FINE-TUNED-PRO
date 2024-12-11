import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperLightMNIST(nn.Module):
    def __init__(self):
        super(SuperLightMNIST, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=0)  # Input: 1x28x28 -> Output: 4x28x28
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=0)  # Input: 4x28x28 -> Output: 8x28x28
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, padding=0)  # Input: 4x14x14 -> Output: 8x14x14
        self.conv4 = nn.Conv2d(8, 12, kernel_size=3, padding=0)  # Input: 8x14x14 -> Output: 12x14x14
        self.conv5 = nn.Conv2d(12, 16, kernel_size=3, padding=0)  # Input: 12x14x14 -> Output: 16x14x14
        self.conv6 = nn.Conv2d(16, 10, kernel_size=3, padding=0)  # Input: 16x3x3 -> Output: 10x1x1


        # Batch normalization layers for each convolutional block
        self.bn1 = nn.BatchNorm2d(4)   # BatchNorm after conv1
        self.bn2 = nn.BatchNorm2d(8)   # BatchNorm after conv2
        self.bn3 = nn.BatchNorm2d(8)   # BatchNorm after conv3
        self.bn4 = nn.BatchNorm2d(12)  # BatchNorm after conv4
        self.bn5 = nn.BatchNorm2d(16)  # BatchNorm after conv5

        # 1D Convolution for channel reduction
        self.conv1d_1 = nn.Conv1d(8, 4, kernel_size=1)  # Input: (batch, 8, 196) -> Output: (batch, 4, 196)
        self.bn1d_1 = nn.BatchNorm1d(4)  # BatchNorm for conv1d_1
        #self.conv1d_2 = nn.Conv1d(16, 8, kernel_size=1)  # Input: (batch, 16, 49) -> Output: (batch, 8, 49)
        #self.bn1d_2 = nn.BatchNorm1d(8)  # BatchNorm for conv1d_2

        # Fully connected layer to map to output classes
        #self.fc1 = nn.Linear(16 * 2 * 2, 10)  # Input: flattened 16x3x3 -> Output: 10 classes

        # Dropout layers for regularization
        self.dropout_1 = nn.Dropout(0.05)  # Dropout with probability 0.05
        self.dropout_2 = nn.Dropout(0.01)  # Dropout with probability 0.01 (not currently used)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)  # Conv1: (batch, 4, 28, 28)
        x = self.bn1(x)    # BatchNorm after conv1
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv2(x)  # Conv2: (batch, 8, 28, 28)
        x = self.bn2(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = F.max_pool2d(x, 2)  # Max pooling: (batch, 8, 14, 14)

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 8, -1)  # Reshape to (batch, 8, 196)

        # First 1D convolution
        x = self.conv1d_1(x)  # Conv1d_1: (batch, 4, 196)
        x = self.bn1d_1(x)    # BatchNorm after conv1d_1
        x = F.relu(x)         # ReLU activation

        # Reshape back to 2D
        x = x.view(batch_size, 4, 12, 12)  # Reshape to (batch, 4, 14, 14)

        # Second convolutional block
        x = self.conv3(x)  # Conv3: (batch, 8, 14, 14)
        x = self.bn3(x)    # BatchNorm after conv3
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv4(x)  # Conv4: (batch, 12, 14, 14)
        x = self.bn4(x)    # BatchNorm after conv4
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv5(x)  # Conv5: (batch, 16, 14, 14)
        x = self.bn5(x)    # BatchNorm after conv5
        x = F.relu(x)      # ReLU activation
        #x = F.max_pool2d(x, 2)  # Max pooling: (batch, 16, 7, 7)

        # Reshape for 1D convolution
        #x = x.view(batch_size, 16, -1)  # Reshape to (batch, 16, 49)

        # Second 1D convolution
        #x = self.conv1d_2(x)  # Conv1d_2: (batch, 8, 49)
        #x = self.bn1d_2(x)    # BatchNorm after conv1d_2
        #x = F.relu(x)         # ReLU activation

        # Reshape back to 2D
        #x = x.view(batch_size, 8, 7, 7)  # Reshape to (batch, 8, 7, 7)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (3, 3))  # Adaptive average pooling: (batch, 8, 3, 3)

        x = self.conv6(x)  # Conv6: (batch, 10, 1, 1)

        # Flatten and fully connected layer
        x = x.view(batch_size, -1)  # Flatten to (batch, 8*3*3)
        #x = self.fc1(x)  # Fully connected layer

        return F.log_softmax(x, dim=1)  # Log-Softmax for classification
    
class SuperLightMNIST_Old(nn.Module):
    def __init__(self):
        super(SuperLightMNIST_Old, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=0)  # Input: 1x28x28 -> Output: 4x28x28
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=0)  # Input: 4x28x28 -> Output: 8x28x28
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, padding=0)  # Input: 4x14x14 -> Output: 8x14x14
        self.conv4 = nn.Conv2d(8, 12, kernel_size=3, padding=0)  # Input: 8x14x14 -> Output: 12x14x14
        self.conv5 = nn.Conv2d(12, 16, kernel_size=3, padding=0)  # Input: 12x14x14 -> Output: 16x14x14


        # Batch normalization layers for each convolutional block
        self.bn1 = nn.BatchNorm2d(4)   # BatchNorm after conv1
        self.bn2 = nn.BatchNorm2d(8)   # BatchNorm after conv2
        self.bn3 = nn.BatchNorm2d(8)   # BatchNorm after conv3
        self.bn4 = nn.BatchNorm2d(12)  # BatchNorm after conv4
        self.bn5 = nn.BatchNorm2d(16)  # BatchNorm after conv5

        # 1D Convolution for channel reduction
        self.conv1d_1 = nn.Conv1d(8, 4, kernel_size=1)  # Input: (batch, 8, 196) -> Output: (batch, 4, 196)
        self.bn1d_1 = nn.BatchNorm1d(4)  # BatchNorm for conv1d_1
        #self.conv1d_2 = nn.Conv1d(16, 8, kernel_size=1)  # Input: (batch, 16, 49) -> Output: (batch, 8, 49)
        #self.bn1d_2 = nn.BatchNorm1d(8)  # BatchNorm for conv1d_2

        # Fully connected layer to map to output classes
        self.fc1 = nn.Linear(16 * 2 * 2, 10)  # Input: flattened 16x3x3 -> Output: 10 classes

        # Dropout layers for regularization
        self.dropout_1 = nn.Dropout(0.05)  # Dropout with probability 0.05
        self.dropout_2 = nn.Dropout(0.01)  # Dropout with probability 0.01 (not currently used)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)  # Conv1: (batch, 4, 28, 28)
        x = self.bn1(x)    # BatchNorm after conv1
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv2(x)  # Conv2: (batch, 8, 28, 28)
        x = self.bn2(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = F.max_pool2d(x, 2)  # Max pooling: (batch, 8, 14, 14)

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 8, -1)  # Reshape to (batch, 8, 196)

        # First 1D convolution
        x = self.conv1d_1(x)  # Conv1d_1: (batch, 4, 196)
        x = self.bn1d_1(x)    # BatchNorm after conv1d_1
        x = F.relu(x)         # ReLU activation

        # Reshape back to 2D
        x = x.view(batch_size, 4, 12, 12)  # Reshape to (batch, 4, 14, 14)

        # Second convolutional block
        x = self.conv3(x)  # Conv3: (batch, 8, 14, 14)
        x = self.bn3(x)    # BatchNorm after conv3
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv4(x)  # Conv4: (batch, 12, 14, 14)
        x = self.bn4(x)    # BatchNorm after conv4
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv5(x)  # Conv5: (batch, 16, 14, 14)
        x = self.bn5(x)    # BatchNorm after conv5
        x = F.relu(x)      # ReLU activation
        #x = F.max_pool2d(x, 2)  # Max pooling: (batch, 16, 7, 7)

        # Reshape for 1D convolution
        #x = x.view(batch_size, 16, -1)  # Reshape to (batch, 16, 49)

        # Second 1D convolution
        #x = self.conv1d_2(x)  # Conv1d_2: (batch, 8, 49)
        #x = self.bn1d_2(x)    # BatchNorm after conv1d_2
        #x = F.relu(x)         # ReLU activation

        # Reshape back to 2D
        #x = x.view(batch_size, 8, 7, 7)  # Reshape to (batch, 8, 7, 7)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (2, 2))  # Adaptive average pooling: (batch, 8, 3, 3)

        # Flatten and fully connected layer
        x = x.view(batch_size, -1)  # Flatten to (batch, 8*3*3)
        x = self.fc1(x)  # Fully connected layer

        return F.log_softmax(x, dim=1)  # Log-Softmax for classification
    
class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # Input: 1x28x28 -> Output: 4x28x28
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # Input: 4x28x28 -> Output: 8x28x28
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # Input: 4x14x14 -> Output: 8x14x14
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Input: 8x14x14 -> Output: 16x14x14
        self.conv5 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Input: 8x7x7 -> Output: 16x7x7
        self.conv6 = nn.Conv2d(16, 24, kernel_size=3, padding=1)  # Input: 16x7x7 -> Output: 24x7x7

        # Batch normalization layers for each convolutional block
        self.bn1 = nn.BatchNorm2d(4)   # BatchNorm after conv1
        self.bn2 = nn.BatchNorm2d(8)   # BatchNorm after conv2
        self.bn3 = nn.BatchNorm2d(8)   # BatchNorm after conv3
        self.bn4 = nn.BatchNorm2d(16)  # BatchNorm after conv4
        self.bn5 = nn.BatchNorm2d(16)  # BatchNorm after conv5
        self.bn6 = nn.BatchNorm2d(24)  # BatchNorm after conv6

        # 1D Convolution for channel reduction
        self.conv1d_1 = nn.Conv1d(8, 4, kernel_size=1)  # Input: (batch, 8, 196) -> Output: (batch, 4, 196)
        self.bn1d_1 = nn.BatchNorm1d(4)  # BatchNorm for conv1d_1
        self.conv1d_2 = nn.Conv1d(16, 8, kernel_size=1)  # Input: (batch, 16, 49) -> Output: (batch, 8, 49)
        self.bn1d_2 = nn.BatchNorm1d(8)  # BatchNorm for conv1d_2

        # Fully connected layer to map to output classes
        self.fc1 = nn.Linear(24 * 3 * 3, 10)  # Input: flattened 24x3x3 -> Output: 10 classes

        # Dropout layers for regularization
        self.dropout_1 = nn.Dropout(0.05)  # Dropout with probability 0.05
        self.dropout_2 = nn.Dropout(0.01)  # Dropout with probability 0.01 (not currently used)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)  # Conv1: (batch, 4, 28, 28)
        x = self.bn1(x)    # BatchNorm after conv1
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv2(x)  # Conv2: (batch, 8, 28, 28)
        x = self.bn2(x)    # BatchNorm after conv2
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = F.max_pool2d(x, 2)  # Max pooling: (batch, 8, 14, 14)

        # Reshape for 1D convolution
        batch_size = x.size(0)
        x = x.view(batch_size, 8, -1)  # Reshape to (batch, 8, 196)

        # First 1D convolution
        x = self.conv1d_1(x)  # Conv1d_1: (batch, 4, 196)
        x = self.bn1d_1(x)    # BatchNorm after conv1d_1
        x = F.relu(x)         # ReLU activation

        # Reshape back to 2D
        x = x.view(batch_size, 4, 14, 14)  # Reshape to (batch, 4, 14, 14)

        # Second convolutional block
        x = self.conv3(x)  # Conv3: (batch, 8, 14, 14)
        x = self.bn3(x)    # BatchNorm after conv3
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = self.conv4(x)  # Conv4: (batch, 16, 14, 14)
        x = self.bn4(x)    # BatchNorm after conv4
        x = F.relu(x)      # ReLU activation
        x = self.dropout_1(x)  # Dropout for regularization
        x = F.max_pool2d(x, 2)  # Max pooling: (batch, 16, 7, 7)

        # Reshape for 1D convolution
        x = x.view(batch_size, 16, -1)  # Reshape to (batch, 16, 49)

        # Second 1D convolution
        x = self.conv1d_2(x)  # Conv1d_2: (batch, 8, 49)
        x = self.bn1d_2(x)    # BatchNorm after conv1d_2
        x = F.relu(x)         # ReLU activation

        # Reshape back to 2D
        x = x.view(batch_size, 8, 7, 7)  # Reshape to (batch, 8, 7, 7)

        # Third convolutional block
        x = self.conv5(x)  # Conv5: (batch, 16, 7, 7)
        x = self.bn5(x)    # BatchNorm after conv5
        x = F.relu(x)      # ReLU activation
        x = self.conv6(x)  # Conv6: (batch, 24, 7, 7)
        x = self.bn6(x)    # BatchNorm after conv6
        x = F.relu(x)      # ReLU activation

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (3, 3))  # Adaptive average pooling: (batch, 24, 3, 3)

        # Flatten and fully connected layer
        x = x.view(batch_size, -1)  # Flatten to (batch, 24*3*3)
        x = self.fc1(x)  # Fully connected layer

        return F.log_softmax(x, dim=1)  # Log-Softmax for classification


class SimpleMNIST(nn.Module):
    """A simpler MNIST model with fewer parameters"""
    def __init__(self):
        super(SimpleMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class DeepMNIST(nn.Module):
    """A deeper MNIST model with more layers"""
    def __init__(self):
        super(DeepMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 3 * 3)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# Dictionary mapping model names to their classes
MODEL_REGISTRY = {
    'light': LightMNIST,
    'simple': SimpleMNIST,
    'deep': DeepMNIST,
    'super_light': SuperLightMNIST
}

def get_model(model_name):
    """Factory function to get a model by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]
