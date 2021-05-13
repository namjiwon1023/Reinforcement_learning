import torch
import torch.nn as nn
import torch.nn.functional as F
from torch..distributions import Categorical
import numpy as np
import gym
import make_env
import math
from buffer import replay_buffer
from net import policy_net, value_net
import os
import time
import copy
import argparse
from gym.spaces.discrete import Discrete

class maddpg(object):
    def __init__(self, env_id, episode, learning_rate, gamma, capacity, batch_size, value_iter, policy_iter, rho, episode_len, render, train_freq, entropy_weight, start_count=10000, model_path=False):
        self.env_id = env_id
        self.env = make_env.make_env(self.env_id)
        self.episode = episode
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.capacity = capacity
        self.batch_size = batch_size
        self.value_iter = value_iter
        self.policy_iter = policy_iter
        self.rho = rho
        self.render = render
        self.episode_len = episode_len
        self.train_freq = train_freq
        self.entropy_weight = entropy_weight
        self.model_path = model_path

        self.observation_dims = self.env.observation_space
        self.action_dims = self.env.action_space

        self.observation_total_dims = sum([self.env.observation_space[i].shape[0] for i in range(self.env.n)])
        self.action_total_dims = sum([self.env.action_space[i].n if isinstance(self.env.action_space[i], Discrete) else sum(self.env.action_space[i].high) + self.env.action_space[i].shape for i in range(self.env.n)])

        self.policy_nets = [policy_net(self.observation_dims[i].shape[0], self.action_dims[i].n if isinstance(self.env.action_space[i], Discrete) else sum(self.env.action_space[i].high) + self.env.action_space[i].shape) for i in range(self.env.n)]
        self.target_policy_nets = [policy_net(self.observation_dims[i].shape[0], self.action_dims[i].n if isinstance(self.env.action_space[i], Discrete) else sum(self.env.action_space[i].high) + self.env.action_space[i].shape) for i in range(self.env.n)]

        self.value_nets = [value_net(self.observation_total_dims, self.action_total_dims, 1) for i in range(self.env.n)]
        self.target_value_nets = [value_net(self.observation_total_dims, self.action_total_dims, 1) for i in range(self.env.n)]

        self.policy_optimizers = [torch.optim.Adam(policy_net.parameters(), lr=self.learning_rate) for policy_net in self.policy_nets]
        self.value_optimizers = [torch.optim.Adam(value_net.parameters(), lr=self.learning_rate) for value_net in self.value_nets]

        if self.model_path:
            for i in range(self.env.n):
                self.policy_nets[i] = torch.load('./models/{}/policy_model{}.pkl'.format(self.env_id, i))
                self.value_nets[i] = torch.load('./models/{}/value_model{}.pkl'.format(self.env_id, i))
        [target_policy_net.load_state_dict(policy_net.state_dict()) for target_policy_net, policy_net in zip(self.target_policy_nets, self.policy_nets)]
        [target_value_net.load_state_dict(value_net.state_dict()) for target_value_net, value_net in zip(self.target_value_nets, self.value_nets)]

        self.buffer = replay_buffer(self.capacity)
        self.count = 0
        self.train_count = 0
        self.start_count = start_count

    def soft_update(self):
        for i in range(self.env.n):
            for param, target_param in zip(self.value_nets[i].parameters(), self.target_value_nets[i].parameters()):
                target_param.detach().copy_(self.rho * target_param.detach() + (1. - self.rho) * param.detach())
            for param, target_param in zip(self.policy_nets[i].parameters(), self.target_policy_nets[i].parameters()):
                target_param.detach().copy_(self.rho * target_param.detach() + (1. - self.rho) * param.detach())

    def train(self):
        for i in range(self.env.n):
            observations, actions, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

            observations_stack = np.vstack([np.hstack(observations[b] for b in range(len(observation)))])
