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
    def __init__(self, in_dims, n_actions, alpha, n_sensors, hidden_size = 256, min_log_std = -20, max_log_std = 2,
                dirPath='/home/nam/Reinforcement_learning/SAC_TORCS_cnn_add_sensors', test_mode=False, with_logprob=True):
        super(ActorNetwork, self).__init__()
        self.test_mode = test_mode
        self.with_logprob = with_logprob
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.checkpoint = os.path.join(dirPath, 'sac_actor')


        self.feature = nn.Sequential(nn.Conv2d(in_dims, 32, kernel_size = 8, stride = 4),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
                                    nn.ReLU())

        self.hidden_layer = nn.Sequential(nn.Linear(64*4*4 + n_sensors, hidden_size),
                                            nn.ReLU())
        # self.hidden_layer_1 = nn.Sequential(nn.Linear(64*4*4 + n_sensors, hidden_size),
        #                                     nn.ReLU())
        # self.hidden_layer_2 = nn.Sequential(nn.Linear(n_sensors, hidden_size),
        #                                     nn.ReLU())
        # self.hidden_layer_3 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
        #                                     nn.ReLU())

        self.mean = nn.Sequential(nn.Linear(hidden_size, n_actions))
        self.log_std = nn.Sequential(nn.Linear(hidden_size, n_actions))


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, sensors):
        feature = self.feature(state)
        feature = feature.view(-1, 64*4*4)
        cat = T.cat((feature, sensors), dim=-1)
        # cat = T.cat((feature, sensors), dim=-1)
        # feature = self.hidden_layer_1(feature)
        # sensors = self.hidden_layer_2(sensors)
        # cat = self.hidden_layer_1(cat)
        cat = self.hidden_layer(cat)

        # cat = feature + sensors
        # cat = self.hidden_layer_3(cat)
        mu = self.mean(cat)
        log_std = self.log_std(cat)
        log_std = T.clamp(log_std, self.min_log_std, self.max_log_std)
        std = T.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()
        if self.test_mode is True:
            action = mu.tanh()
        else:
            action = z.tanh()
        if self.with_logprob:
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