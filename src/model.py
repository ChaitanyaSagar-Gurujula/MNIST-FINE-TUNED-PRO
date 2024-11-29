import torch
import torch.nn as nn
import torch.nn.functional as F

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        # Slightly increase initial channels but keep params under 25k
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 8x28x28
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 16x14x14

        # 1D Convolutions for channel reduction
        self.conv1d_1 = nn.Conv1d(8, 6, kernel_size=1)  # Reduce channels from 16 to 12
        self.conv1d_2 = nn.Conv1d(12, 8, kernel_size=1)   # Further reduce to 8
        
        self.fc1 = nn.Linear(16 * 7 * 7, 10)
        
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.05)  # Reduced dropout slightly
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)  # Add dropout after relu
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = self.dropout(x)  # Add dropout after relu
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1) 