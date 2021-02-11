import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNet(nn.Module):
    def __init__(self, in_dim, action_size):
        super(CriticNet, self).__init__()
        self.conv = nn.Conv2d(in_dim, 10, kernel_size = 2, stride = 1, padding = 0)
        self.pool = nn.AvgPool2d((2,1))
        self.fc = nn.Linear(30, 6)

    def forward(self,state):
        x = self.pool(F.relu(self.conv(state)))
        out = self.fc(x)
        return out