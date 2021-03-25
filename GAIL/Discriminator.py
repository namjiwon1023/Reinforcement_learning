import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Discriminator(nn.Module):
    def __init__(self, n_states, n_actions, alpha, hidden_size=128, dirPath='/home/nam/Reinforcement_learning/GAIL'):
        super(Discriminator, self).__init__()
        self.checkpoint = os.path.join(dirPath, 'sac_discriminator')
        self.Discriminator = nn.Sequential(nn.Linear(n_states + n_actions, hidden_size),
                                            nn.Tanh(),
                                            nn.Linear(hidden_size, hidden_size),
                                            nn.Tanh(),
                                            nn.Linear(hidden_size, 1),
                                            nn.Sigmoid())
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.Discriminator_loss = F.BCELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_action = T.cat((state, action), dim=1)
        prob = self.Discriminator(state_action)
        return prob

    def save_models(self):
        T.save(self.state_dict(), self.checkpoint)

    def load_models(self):
        self.load_state_dict(T.load(self.checkpoint))

