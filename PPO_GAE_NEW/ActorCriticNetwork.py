import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size, alpha, dirPath='/home/nam/Reinforcement_learning/PPO_GAE_NEW'):
        super(ActorNetwork, self).__init__()
        self.feature = nn.Sequential(nn.Linear(n_states, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU())
        self.mean = nn.Sequential(nn.Linear(hidden_size, n_actions))
        self.log_std = nn.Sequential(nn.Linear(hidden_size, n_actions))

        self.checkpoint = os.path.join(dirPath, 'actor_ppo')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        feature = self.feature(state)
        mu = self.mean(feature)
        log_std = self.log_std(feature)
        std = T.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist

    def save_models(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_models(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))


class CriticNetwork(nn.Module):
    def __init__(self, n_states, hidden_size, alpha, dirPath='/home/nam/Reinforcement_learning/PPO_GAE_NEW'):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(nn.Linear(n_states, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
        self.checkpoint = os.path.join(dirPath, 'critic_ppo')
        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_models(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_models(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))

