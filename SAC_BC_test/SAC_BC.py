
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
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

if T.backends.cudnn.enabled:
    T.backends.cudnn.benchmark = False
    T.backends.cudnn.deterministic = True

seed = 777
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

class SACAgent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # self.env = gym.make('Pendulum-v0')
        self.env = gym.make('BipedalWalker-v2')
        # self.env = ActionNormalizer(self.env)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.memory = ReplayBuffer(self.memory_size, self.n_states, self.batch_size)

        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.actor = ActorNetwork(self.n_states, self.n_actions, self.max_action, self.learning_rate)

        self.critic_eval = CriticNetwork(self.n_states, self.n_actions, self.learning_rate)
        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.transition = list()

        self.total_step = 0

        # BC loss
        lambda1 = 1e-3
        lambda2 = 1.0
        self.lambda1 = lambda1
        self.lambda2 = lambda2 / self.store_memory_batch_size
        self.store_memory_batch_size = 128
        self.store_memory = np.load('store_memory.npy')

        self.memory_backup = ReplayBuffer(len(self.store_memory), self.n_states, self.store_memory_batch_size)
        self.memory_backup.store_with_store_memory(self.store_memory)

    def choose_action(self, state, test_mode):
        test_mode = self.test_mode
        if (self.total_step < self.train_start) and not test_mode:
            action = self.env.action_space.sample()
        if self.test_mode is True:
            action = self.actor(T.FloatTensor(state).to(self.actor.device), test_mode=True, with_logprob=False)[0].detach().cpu().numpy()
        else:
            action = self.actor(T.FloatTensor(state).to(self.actor.device), test_mode=False, with_logprob=True)[0].detach().cpu().numpy()
        self.transition = [state, action]
        return action

    def target_soft_update(self):
        tau = self.tau
        with T.no_grad():
            for t_p, l_p in zip(self.critic_target.parameters(), self.critic_eval.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def learn(self):
        samples = self.memory.sample_batch()
        state = T.FloatTensor(samples["state"]).to(self.actor.device)
        next_state = T.FloatTensor(samples["next_state"]).to(self.actor.device)
        action = T.FloatTensor(samples["action"].reshape(-1, 1)).to(self.actor.device)
        reward = T.FloatTensor(samples["reward"].reshape(-1, 1)).to(self.actor.device)
        done = T.FloatTensor(samples["done"].reshape(-1, 1)).to(self.actor.device)
        mask = (1 - done).to(self.actor.device)


        samples_b = self.memory_backup.sample_batch()
        state_b = T.FloatTensor(samples_b["state"]).to(self.actor.device)
        next_state_b = T.FloatTensor(samples_b["next_state"]).to(self.actor.device)
        action_b = T.FloatTensor(samples_b["action"].reshape(-1, 1)).to(self.actor.device)
        reward_b = T.FloatTensor(samples_b["reward"].reshape(-1, 1)).to(self.actor.device)
        done = T.FloatTensor(samples_b["done"].reshape(-1, 1)).to(self.actor.device)


        # critic update
        with T.no_grad():
            next_action, next_log_prob = self.actor(next_state, test_mode=False, with_logprob=True)
            q1_target, q2_target = self.critic_target(next_state, next_action)
            q_target = T.min(q1_target, q2_target)
            value_target = reward + self.GAMMA * (q_target - self.alpha * next_log_prob) # action value function(Q_soft)
        q1_eval, q2_eval = self.critic_eval(state, action)
        critic_loss = F.mse_loss(q1_eval, value_target) + F.mse_loss(q2_eval, value_target)

        self.critic_eval.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval.optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = False
        # actor and alpha update
        new_action, new_log_prob = self.actor(state, test_mode=False, with_logprob=True)
        q_1, q_2 = self.critic_eval(state, new_action)
        q = T.min(q_1, q_2)
        pg_loss = (self.alpha * new_log_prob - q).mean()      # actor loss function
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()

        pred_action, _ = self.actor(state_b, test_mode=False, with_logprob=True)
        qf_mask = T.gt(self.critic_eval(state_b, action_b),self.critic_eval(state_b, pred_action),).to(self.critic_eval.device)
        qf_mask = qf_mask.float()
        n_qf_mask = int(qf_mask.sum().item())

        if n_qf_mask == 0:
            bc_loss = T.zeros(1, device=device)
        else:
            bc_loss = (
                T.mul(pred_action, qf_mask) - T.mul(action_b, qf_mask)
            ).pow(2).sum() / n_qf_mask

        actor_loss = self.lambda1 * pg_loss + self.lambda2 * bc_loss

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = True

        self.alpha = self.log_alpha.exp()

        if self.total_step % self.update_time == 0 :
            self.target_soft_update()
