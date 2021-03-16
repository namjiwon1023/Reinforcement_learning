import torch as T
import torch.nn as nn
import torch.optim as optim
import os

class CriticQ(nn.Module):
    def __init__(self, n_states):
        super(CriticQ, self).__init__()
        self.criticQ_1 = nn.Sequential(nn.Linear(n_states, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 1))
        self.criticQ_2 = nn.Sequential(nn.Linear(n_states, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 1))
    def forward(self, state, action):
        x = T.cat((state, action), dim=-1)
        Q1 = self.criticQ_1(x)
        Q2 = self.criticQ_2(x)
        return Q1. Q2

class CriticV(nn.Module):
    def __init__(self, n_states):
        super(CriticV, self).__init__()
        self.criticV = nn.Sequential(nn.Linear(n_states, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 1))
    def forward(self, state):
        value = self.criticV(state)
        return value