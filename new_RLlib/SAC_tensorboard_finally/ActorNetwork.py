import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, alpha, dirPath, min_log_std = -20, max_log_std = 2):
        super(ActorNetwork, self).__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.checkpoint = os.path.join(dirPath, 'sac_actor')

        self.feature = nn.Sequential(nn.Linear(n_states, n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden, n_hidden),
                                    nn.ReLU(),
                                    )

        self.log_std = nn.Linear(n_hidden, n_actions)

        self.mu = nn.Linear(n_hidden, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, test_mode=False, with_logprob=True):

        feature = self.feature(state)

        mu = self.mu(feature)
        log_std = self.log_std(feature)
        log_std = T.clamp(log_std, self.min_log_std, self.max_log_std)
        std = T.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        if test_mode is True:
            action = mu.tanh()
        else:
            action = z.tanh()

        if with_logprob is True:
            log_prob = dist.log_prob(z) - T.log(1 - action.pow(2) + 1e-7)
            log_prob = log_prob.sum(-1, keepdim=True)
        else:
            log_prob = None

        return action, log_prob

    def save_models(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_models(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))
