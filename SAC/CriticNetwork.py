import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os

class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, alpha,
                    dirPath='/home/nam/Reinforcement_learning/SAC'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_path = os.path.join(dirPath, 'sac_critic')

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.q1 = nn.Sequential(nn.Linear(n_states + n_actions, n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden, n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden, 1))

        self.q2 = nn.Sequential(nn.Linear(n_states + n_actions, n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden, n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden, 1))


    def forward(self, state, action):
        cat = T.cat([state, action], 1)
        Q1 = self.q1(cat)
        return Q1

    def get_double_Q(self, state, action):
        cat = T.cat([state, action], 1)
        Q1 = self.q1(cat)
        Q2 = self.q2(cat)
        return Q1, Q2

    def save_models(self):
        T.save(self.state_dict(), self.checkpoint_path)
        T.save(self.optimizer.state_dict(), self.checkpoint_path + '_optimizer')

    def load_models(self):
        self.load_state_dict(T.load(self.checkpoint_path))
        self.optimizer.load_state_dict(T.load(self.checkpoint_path + '_optimizer'))