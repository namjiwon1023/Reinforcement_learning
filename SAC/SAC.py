import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import gym
import copy

from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class SACAgent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.env = gym.make('Pendulum-v0')

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.critic_eval = CriticNetwork(self.n_states, self.n_actions, self.n_hidden, self.critic_lr)
        self.critic_target = copy.deepcopy(self.critic_eval)

        self.actor_net = ActorNetwork(self.n_states, self.n_actions, self.n_hidden, self.min_log_std, self.max_log_std, self.actor_lr)

        self.target_entropy = -np.prod(self.env.action_space.shape).item()
        self.log_alpha = T.zeros(1, requires_grad=True, device=device)

        self.alpha_optim = T.optim.Adam([self.log_alpha], lr=self.actor_lr)

        self.memory = ReplayBuffer(self.memory_size, self.n_states, self.batch_size)

