import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import gym

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from GaussianNoise import GaussianNoise


def _layer_norm(layer, std=1.0, bias_const=1e-6):
    if type(layer) == nn.Linear:
        T.nn.init.orthogonal_(layer.weight, std)
        T.nn.init.constant_(layer.bias, bias_const)

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

class TD3Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.env = gym.make('Pendulum-v0')
        self.n_states =  self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.env = ActionNormalizer(self.env)

        self.actor_eval = ActorNetwork(self.n_states, self.n_actions, self.actor_lr)
        self.actor_target = copy.deepcopy(self.actor_eval)

        self.critic_eval = CriticNetwork(self.n_states, self.n_actions, self.critic_lr)
        self.critic_target = copy.deepcopy(self.critic_eval)

        self.memory = ReplayBuffer(self.memory_size, self.n_states)
        self.transition = list()

        self.exploration_noise = GaussianNoise(self.n_actions, self.exploration_noise, self.exploration_noise)
        self.target_policy_noise = GaussianNoise(self.n_actions, self.policy_noise, self.policy_noise)


    def choose_action(self, state, n_actions):
        # s = T.unsqueeze(T.FloatTensor(state),0).to(self.actor_eval.device)

        if self.total_episode < self.train_start:
            # action = np.random.randint(0, n_actions)
            action = self.env.action_space.sample()
        else:
            # action = self.actor_eval(s).detach().cpu().numpy()
            action = self.actor_eval(T.FloatTensor(state).to(self.actor_eval.device))[0].detach().cpu().numpy()
            noise = self.exploration_noise.sample()
            action = np.clip(action + noise, -1.0, 1.0)
        self.transition = [state, action]
        return action

    def learn(self):
        self.learn_step += 1

        samples = self.memory.sample_batch()
        state = T.FloatTensor(samples['state']).to(self.actor_eval.device)
        action = T.FloatTensor(samples['action'].reshape(-1, 1)).to(self.actor_eval.device)
        reward = T.FloatTensor(samples['reward'].reshape(-1, 1)).to(self.actor_eval.device)
        next_state = T.FloatTensor(samples['next_state']).to(self.actor_eval.device)
        done = T.FloatTensor(samples['done'].reshape(-1, 1)).to(self.actor_eval.device)

        mask = (1 - done).to(self.actor_eval.device)

        with T.no_grad():
            noise = (T.FloatTensor(self.target_policy_noise.sample()).to(self.actor_eval.device))
            clip_noise = T.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + clip_noise).clamp(-1.0, 1.0)

            next_target_Q1, next_target_Q2 = self.critic_target.get_double_Q(next_state, next_action)
            next_target_Q = T.min(next_target_Q1, next_target_Q2)
            target_Q = reward + self.GAMMA*next_target_Q*mask

        current_Q1, current_Q2 = self.critic_eval.get_double_Q(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_eval.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval.optimizer.step()

        if self.learn_step % self.update_time == 0:
            actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()

            self.actor_eval.optimizer.zero_grad()
            actor_loss.backward()
            self.actor_eval.optimizer.step()

            self.targetNet_soft_update()


    def targetNet_soft_update(self):

        for eval_params, target_params in zip(self.critic_eval.parameters(), self.critic_target.parameters()):
            target_params.data.copy_(self.tau * eval_params.data + (1 - self.tau) * target_params.data)

        for eval_params, target_params in zip(self.actor_eval.parameters(), self.actor_target.parameters()):
            target_params.data.copy_(self.tau * eval_params.data + (1 - self.tau) * target_params.data)

    def save_models(self):
        self.actor_eval.save_models()
        self.critic_eval.save_models()

    def load_models(self):
        self.actor_eval.load_models()
        self.actor_target = copy.deepcopy(self.actor_eval)

        self.critic_eval.load_models()
        self.critic_target = copy.deepcopy(self.critic_eval)


