import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, alpha, device):
        super(CriticNetwork, self).__init__()

        self.criticQ_1 = nn.Sequential(nn.Linear(n_states + n_actions, n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(n_hidden, n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(n_hidden, 1))

        self.criticQ_2 = nn.Sequential(nn.Linear(n_states + n_actions, n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(n_hidden, n_hidden),
                                        nn.ReLU(),
                                        nn.Linear(n_hidden, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat((state, action), dim=-1)

        Q1 = self.criticQ_1(x)
        Q2 = self.criticQ_2(x)

        return Q1, Q2
