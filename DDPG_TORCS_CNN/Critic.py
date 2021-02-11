import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(state_size,8,kernel_size = 4 , stride = 2 , padding = 2)
        self.b1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size = 4, stride = 1, padding = 2)
        self.b2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size = 4, stride = 1, padding = 2)
        self.b3 = nn.BatchNorm2d(32)

        self.state = nn.Linear(39200, 2048)

        self.action = nn.Linear(action_size, 2048)

        self.hidden_layer_1 = nn.Linear(2048, 2048)
        self.hidden_layer_2 = nn.Linear(2048, 512)
        self.action_value = nn.Linear(512, action_size)

    def forward(self, state, action):
        x = F.relu(self.b1(self.conv1(state)))
        x = F.relu(self.b2(self.conv2(x)))
        x = F.relu(self.b3(self.conv3(x)))
        x = x.view(x.size(0),-1)

        state_v = self.state(x)
        action_v = self.action(action)
        cat = state_v + action_v
        hidden_1 = self.hidden_layer_1(cat)
        hidden_2 = self.hidden_layer_2(hidden_1)
        V = self.action_value(hidden_2)

        return V