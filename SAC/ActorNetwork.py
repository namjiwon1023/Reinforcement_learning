import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, min_log_std = -20, max_log_std = 2,):
        super(Actor, self).__init__()

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.feature = nn.Sequential(nn.Linear(n_states, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 128),
                                    nn.ReLU())

        self.log_std = nn.Linear(128, n_actions)

        self.avg = nn.Linear(128, n_actions)

    def forward(self, state):

        feature = self.feature(state)

        avg = self.avg(feature).tanh()

        log_std = self.log_std(feature).tanh()
        log_std = self.min_log_std + 0.5 * (self.max_log_std - self.min_log_std) * (log_std + 1)
        std = T.exp(log_std)

        dist = Normal(avg, std)
        z = dist.rsample()

        action = z.tanh()
        log_prob = dist.log_prob(z) - T.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob
