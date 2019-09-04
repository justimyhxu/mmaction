import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class EncodePosition(nn.Module):
    def __init__(self, in_dim = 60, h_dim=512, num_classes=10, init_std=0.01):
        super(EncodePosition, self).__init__()
        self.h_dim = h_dim
        self.num_classes = num_classes
        self.init_std = init_std
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, num_classes)
        relu = nn.ReLU(inplace=True)

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, 0, self.init_std)
        nn.init.constant_(self.fc1.bias, 0)

        nn.init.normal_(self.fc2.weight, 0, self.init_std)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x  = relu(x)
        x  = self.fc2(x)
        return x
