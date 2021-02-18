import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_size, 32, kernel_size = 8, stride = 4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        self.state = nn.Linear(64*5*5, 512)

        self.action = nn.Linear(action_size, 512)

        self.hidden_layer_1 = nn.Linear(512, 512)
        self.action_value = nn.Linear(512, action_size)

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*5*5)

        state_v = F.relu(self.state(x))
        action_v = F.relu(self.action(action))
        cat = state_v + action_v
        hidden_1 = F.relu(self.hidden_layer_1(cat))
        V = self.action_value(hidden_1)

        return V