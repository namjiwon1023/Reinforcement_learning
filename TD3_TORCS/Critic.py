import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_1 = 300, hidden_layer_2 = 600):
        super(Critic, self).__init__()
        self.state_1 = nn.Linear(state_size, hidden_layer_1)
        self.state_2 = nn.Linear(hidden_layer_1, hidden_layer_2)

        self.action_1 = nn.Linear(action_size, hidden_layer_2)

        self.hidden_layer = nn.Linear(hidden_layer_2, hidden_layer_2)

        self.state_value = nn.Linear(hidden_layer_2, action_size)

    def forward(self, state, action):
        state_hidden_1 = F.relu(self.state_1(state))
        action_hidden_2 = self.action_1(action)
        state_hidden_2 = self.state_2(state_hidden_1)
        cat = state_hidden_2 + action_hidden_2
        hidden = F.relu(self.hidden_layer(cat))
        V = self.state_value(hidden)
        return V