import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, log_std_min = -20, log_std_max = 2,):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)

        self.log_std_layer = nn.Linear(128, out_dim)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

        self.mu_layer = nn.Linear(128, out_dim)
        self.mu_layer = init_layer_uniform(self.mu_layer)

    def forward(self, state):

        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))

        mu = self.mu_layer(x).tanh()

        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = T.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob
