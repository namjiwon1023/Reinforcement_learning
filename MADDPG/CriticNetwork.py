import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hiddens, alpha, device):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(nn.Linear(n_states + n_actions, n_hiddens),
                                nn.ELU(),
                                nn.Linear(n_hiddens, n_hiddens),
                                nn.ELU(),
                                nn.Linear(n_hiddens, n_hiddens),
                                nn.ELU(),
                                nn.Linear(n_hiddens, 1))

        self.optimizer = optim.AdamW(self.parameters(),lr=alpha)
        self.to(device)

    def forward(self, state, action):
        data = T.cat([state, action], dim=1)
        value = self.critic(data)

        return value
