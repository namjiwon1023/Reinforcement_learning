import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hiddens, alpha, device):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(nn.Linear(n_states, n_hiddens),
                                nn.ELU(),
                                nn.Linear(n_hiddens, n_hiddens),
                                nn.ELU(),
                                nn.Linear(n_hiddens, n_hiddens),
                                nn.ELU(),
                                nn.Linear(n_hiddens, n_actions))

        self.optimizer = optim.AdamW(self.parameters(),lr=alpha)
        self.to(device)

    def forward(self, state):
        x = self.actor(state)
        action = T.tanh(x)

        return action
