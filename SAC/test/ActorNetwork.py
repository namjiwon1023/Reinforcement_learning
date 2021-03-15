import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, min_log_std = -20, max_log_std = 2,):
        super(Actor, self).__init__()

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        self.log_std = nn.Linear(128, out_dim)

        self.avg = nn.Linear(128, out_dim)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        avg = self.avg(x).tanh()

        log_std = self.log_std(x).tanh()
        log_std = self.min_log_std + 0.5 * (self.max_log_std - self.min_log_std) * (log_std + 1)
        std = T.exp(log_std)

        dist = Normal(avg, std)
        z = dist.rsample()

        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob
