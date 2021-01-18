import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_uniformly(layer = nn.Linear, init_w = 3e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Critic(nn.Module):
    def __init__(self, state_size, hidden_layer_1 = 300, hidden_layer_2 = 600):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.out = nn.Linear(hidden_layer_2, 1)

        initialize_uniformly(self.out)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        V = self.out(x)
        return V