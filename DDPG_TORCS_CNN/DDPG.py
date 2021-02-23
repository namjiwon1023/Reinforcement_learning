import copy
import random

from gym_torcs import TorcsEnv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

import matplotlib.pyplot as plt

from Actor import Actor
from Critic import Critic
from OU import OU
from ReplayBuffer import ReplayBuffer

EPISODE_COUNT = 50000
MAX_STEPS = 100000
EXPLORE = 5000000.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # memory setting
        self.memory = ReplayBuffer(self.state_size, self.memory_size, self.batch_size)
        self.transition = list()

        # Actor Network setting(eval , target)
        self.actor_eval = Actor(self.state_size).to(device)

        self.actor_target = Actor(self.state_size).to(device)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.actor_target.eval()

        # Critic Network setting(eval , target)
        self.critic_eval = Critic(self.state_size, self.action_size).to(device)

        self.critic_target = Critic(self.state_size, self.action_size).to(device)
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.critic_target.eval()

        # Optimization Setting
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.critic_lr)

        # Loss Function
        self.critic_loss_func = nn.MSELoss()

        self.critic_L = 0.
        self.actor_L = 0.

        # Load Neural Network Parameters
        self.dirPath = './load_state/DDPG_epsilon_'
        self.Actor_dirPath = './load_state/DDPG_actor_'
        self.Critic_dirPath = './load_state/DDPG_critic_'

        if self.load_model :
            self.actor_eval.load_state_dict(torch.load(self.Actor_dirPath + str(self.load_episode) + '.h5'))
            self.critic_eval.load_state_dict(torch.load(self.Critic_dirPath + str(self.load_episode) + '.h5'))
            with open(self.dirPath + str(self.load_episode) + '.pkl','rb') as outfile:
                param = pickle.load(outfile)
                if param is not None:
                    self.epsilon = param.get('epsilon')
                else:
                    pass

    def learn(self):
        # Random Samples
        samples = self.memory.sample_batch()

        state = torch.FloatTensor(samples['obs']).to(device)
        next_state = torch.FloatTensor(samples['next_obs']).to(device)
        action = torch.FloatTensor(samples['act']).reshape(-1,3).to(device)
        reward = torch.FloatTensor(samples['rew']).reshape(-1,1).to(device)
        done = torch.FloatTensor(samples['done']).reshape(-1,1).to(device)

        # Critic Network Update
        mask = (1 - done).to(device)
        next_action = self.actor_target(next_state).to(device)
        next_value = self.critic_target(next_state, next_action).to(device)
        target_values = (reward + self.gamma * next_value * mask).to(device)

        eval_values = self.critic_eval(state, action).to(device)
        critic_loss = self.critic_loss_func(eval_values, target_values).to(device)
        self.critic_L = critic_loss.detach().cpu().numpy()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Network Update
        actor_loss = -self.critic_eval(state, self.actor_eval(state)).to(device).mean()
        self.actor_L = actor_loss.detach().cpu().numpy()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update -> Target Network Parameters
        # θQ‘←τθQ+(1−τ)θQ‘
        # θμ‘←τθμ+(1−τ)θμ‘
        self._target_soft_update()

    def _target_soft_update(self):
        tau = self.tau

        for target_params , eval_params in zip(
            self.actor_target.parameters(), self.actor_eval.parameters()
            ):
            target_params.data.copy_(tau * eval_params.data + (1.0 - tau) * target_params.data)

        for target_params , eval_params in zip(
            self.critic_target.parameters(), self.critic_eval.parameters()
            ):
            target_params.data.copy_(tau * eval_params.data + (1.0 - tau) * target_params.data)
