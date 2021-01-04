import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_1 = nn.Linear(state_size, 300)
        self.state_2 = nn.Linear(300, 600)

        self.action_1 = nn.Linear(action_size, 600)

        self.hidden_layer = nn.Linear(600, 600)

        self.state_value = nn.Linear(600, action_size)

    def forward(self, state, action):
        s_300 = F.relu(self.state_1(state))
        a_600 = self.action_1(action)
        s_600 = self.state_2(s_300)
        cat = s_600 + a_600
        hidden = F.relu(self.hidden_layer(cat))
        V = self.state_value(hidden)
        return V