import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, in_dims, n_actions, alpha, hidden_size_1=256, hidden_size_2=512, dirPath='/home/nam/Reinforcement_learning/SAC_TORCS_cnn'):
        super(CriticNetwork, self).__init__()
        self.checkpoint = os.path.join(dirPath, 'sac_critic')

        self.feature = nn.Sequential(nn.Conv2d(in_dims, 32, kernel_size = 8, stride = 4),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
                                    nn.ReLU())

        self.hidden_layer_1 = nn.Sequential(nn.Linear(64*4*4, hidden_size_1),
                                            nn.ReLU())

        self.hidden_layer_2 = nn.Sequential(nn.Linear(n_actions, hidden_size_1),
                                            nn.ReLU())

        self.criticQ_1 = nn.Sequential(nn.Linear(hidden_size_1, hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size_2, hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size_2, 1))

        self.criticQ_2 = nn.Sequential(nn.Linear(hidden_size_1, hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size_2, hidden_size_2),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size_2, 1))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        feature = self.feature(state)
        feature = feature.view(-1, 64*4*4)
        state_feature = self.hidden_layer_1(feature)

        action_feature = self.hidden_layer_2(action)

        cat = state_feature + action_feature

        Q1 = self.criticQ_1(cat)
        Q2 = self.criticQ_2(cat)

        return Q1, Q2

    def save_models(self):
        T.save(self.state_dict(), self.checkpoint)
        T.save(self.optimizer.state_dict(), self.checkpoint + '_optimizer')

    def load_models(self):
        self.load_state_dict(T.load(self.checkpoint))
        self.optimizer.load_state_dict(T.load(self.checkpoint + '_optimizer'))