import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCriticCNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticCNN, self).__init__()
        self.conv1 = nn.Conv2d(state_size, 32, kernel_size = 8, stride = 4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)       # 64 -> 128 / 64 -> 64

        self.fc1 = nn.Linear(64*5*5, 512)

        self.fc2 = nn.Linear(512, action_size)

    def forward(self, s):
        x = F.relu(self.conv1(s))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*5*5)

        x = F.relu(self.fc1(x))

        out = self.fc2(x)
        return out

