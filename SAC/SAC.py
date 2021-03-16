
import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import gym
import random

from ReplayBuffer import ReplayBuffer
from ActorNetwork import Actor
from CriticNetwork import CriticQ, CriticV

if T.backends.cudnn.enabled:
    T.backends.cudnn.benchmark = False
    T.backends.cudnn.deterministic = True

seed = 123
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

# def _layer_norm(layer, std=1.0, bias_const=1e-6):
#     if type(layer) == nn.Linear:
#         T.nn.init.orthogonal_(layer.weight, std)
#         T.nn.init.constant_(layer.bias, bias_const)

'''OpenAI Gym '''
class ActionNormalizer(gym.ActionWrapper):
    def action(self, action: np.ndarray) -> np.ndarray:
        low = self.action_space.low
        high = self.action_space.high
        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor
        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)
        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        low = self.action_space.low
        high = self.action_space.high
        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor
        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)
        return action


class SACAgent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.env = gym.make('Pendulum-v0')
        # self.env = ActionNormalizer(self.env)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.memory = ReplayBuffer(self.memory_size, self.n_states, self.batch_size)

        self.target_entropy = -np.prod((self.n_actions,)).item()
        self.log_alpha = T.zeros(1, requires_grad=True, device=device)

        self.actor = Actor(self.n_states, self.n_actions).to(device)

        self.vf = CriticV(self.n_states).to(device)
        self.vf_target = copy.deepcopy(self.vf)

        self.qf = CriticQ(self.n_states + self.n_actions).to(device)

        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.learning_rate)
        self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=self.learning_rate)

        self.transition = list()

        self.total_step = 0

    def choose_action(self, state):
        if (self.total_step < self.train_start) and not self.test_mode:
            action = self.env.action_space.sample()
        else:
            action = self.actor(T.FloatTensor(state).to(device))[0].detach().cpu().numpy()
        self.transition = [state, action]
        return action

    def target_soft_update(self):
        tau = self.tau
        for t_p, l_p in zip(self.vf_target.parameters(), self.vf.parameters()):
            t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def learn(self):
        samples = self.memory.sample_batch()
        state = T.FloatTensor(samples["state"]).to(device)
        next_state = T.FloatTensor(samples["next_state"]).to(device)
        action = T.FloatTensor(samples["action"].reshape(-1, 1)).to(device)
        reward = T.FloatTensor(samples["reward"].reshape(-1, 1)).to(device)
        done = T.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        new_action, log_prob = self.actor(state)

        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        mask = 1 - done
        q_1_pred, q_2_pred = self.qf(state, action)
        v_target = self.vf_target(next_state)
        q_target = reward + self.GAMMA * v_target * mask
        qf_loss = F.mse_loss(q_1_pred, q_target.detach()) + F.mse_loss(q_2_pred, q_target.detach())

        v_pred = self.vf(state)
        q_pred = T.min(*self.qf(state, new_action))

        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_step % self.update_time == 0 :
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.target_soft_update()
        else:
            actor_loss = T.zeros(1)

        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()