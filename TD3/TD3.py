import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from GaussianNoise import GaussianNoise

def _layer_norm(layer, std=1.0, bias_const=1e-6):
    if type(layer) == nn.Linear:
        T.nn.init.orthogonal_(layer.weight, std)
        T.nn.init.constant_(layer.bias, bias_const)

class TD3Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.actor_eval = ActorNetwork(self.n_states, self.n_actions, self.lr)
        self.actor_target = copy.deepcopy(self.actor_eval)

        self.critic_eval = CriticNetwork(self.n_states, self.n_actions, self.lr)
        self.critic_target = copy.deepcopy(self.critic_eval)

        self.memory = ReplayBuffer(self.memory_size, self.n_states, self.n_actions)
        self.transition = list()

    def choose_action(self, state):
        s = T.unsqueeze(T.FloatTensor(state),0).to(self.actor_eval.device)
        action = self.actor_eval(s).detach().cpu().numpy()
        self.transition = [state, action]
        return action

    def learn(self):
        self.learn_step += 1

        samples = self.memory.sample_batch()
        state = T.FloatTensor(samples['state']).to(self.actor.device)
        action = T.Floattensor(samples['action']).to(self.actor.device)
        reward = T.FloatTensor(samples['reward']).to(self.actor.device)
        next_state = T.FloatTensor(samples['next_state']).to(self.actor.device)
        done = T.FloatTensor(samples['done']).to(self.actor.device)

        mask = (1 - done).to(self.actor.device)

        with T.no_grad():
            noise = (T.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, slef.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-2, 2)

            next_target_Q1, next_target_Q2 = self.critic_target.get_double_Q(next_state, next_action)
            next_target_Q = T.min(next_target_Q1, next_target_Q2)
            target_Q = reward + self.gamma *next_target_Q*mask

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



if __name__ == '__main__':
    params = {
                'n_states' : 29,
                'n_actions' : 3,
                'GAMMA' : 0.99,
                'tau' : 0.005,
                'policy_noise' : 0.2,
                'noise_clip' : 0.5,
                'lr' : 3e-4,
                'update_time' : 2,
                'memory_size' : 100000,
                'batch_size' : 64,
                'learn_step' : 0
}