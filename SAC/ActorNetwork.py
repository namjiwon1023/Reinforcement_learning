import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, min_log_std, max_log_std, alpha,
                    dirPath='/home/nam/Reinforcement_learning/SAC'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_path = os.path.join(dirPath, 'sac_actor')

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.mlp1 = nn.Sequential(nn.Linear(n_states, n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden, n_hidden),
                                    nn.ReLU())

        self.avg = nn.Sequential(nn.Linear(n_hidden, n_actions))
        self.log_std = nn.Sequential(nn.Linear(n_hidden, n_actions))

        self.max_log_std = max_log_std
        self.min_log_std = min_log_std

    def forward(self, state):
        feature = self.mlp1(state)

        avg = self.avg(feature)
        log_std = self.log_std(feature)
        log_std = T.clamp(log_std, min=self.min_log_std, max=self.max_log_std)

        return (avg, T.exp(log_std))

    def save_models(self):
        T.save(self.state_dict(), self.checkpoint_path)
        T.save(self.optimizer.state_dict(), self.checkpoint_path + '_optimizer')

    def load_models(self):
        self.load_state_dict(T.load(self.checkpoint_path))
        self.optimizer.load_state_dict(T.load(self.checkpoint_path + '_optimizer'))