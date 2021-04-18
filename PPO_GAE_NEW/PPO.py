import numpy as np
import torch as T
import random
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque
import gym

from ActorCriticNetwork import ActorNetwork, CriticNetwork

def compute_gae(next_value, rewards, masks, values, gamma, tau):
    values = values + [next_value]
    gae = 0
    returns = deque()
    for step in reversed(range(len(rewards))):
        delta = (rewards[step] + gamma * values[step + 1] * masks[step] - values[step])
        gae = delta + gamma * tau * masks[step] * gae
        returns.appendleft(gae + values[step])

    return list(returns)

def ppo_iter(epoch, mini_batch_size, states, actions, values, log_probs, returns, advantages):
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[rand_ids], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.env = gym.make('Pendulum-v0')

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.hidden_size = 128

        self.actor = ActorNetwork(self.n_states, self.n_actions, self.hidden_size, self.lr_actor)
        self.critic = CriticNetwork(self.n_states, self.hidden_size, self.lr_critic)

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []
    def choose_action(self, state):
        state = T.FloatTensor(state).to(self.actor.device)
        action, dist = self.actor(state)
        if self.test_mode :
            choose_action = dist.mean
        else:
            choose_action = action
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(choose_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(choose_action))
        return choose_action.cpu().detach().numpy()


    def learn(self, next_state):
        next_state = T.FloatTensor(next_state).to(self.critic.device)
        next_value = self.critic(next_state)

        returns = compute_gae(next_value, self.rewards, self.masks, self.values, self.gamma, self.tau,)

        states = T.cat(self.states).view(-1, 3)
        actions = T.cat(self.actions)
        returns = T.cat(returns).detach()
        values = T.cat(self.values).detach()
        log_probs = T.cat(self.log_probs).detach()
        advantages = returns - values

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(epoch = self.epoch,
                                                                            mini_batch_size = self.batch_size,
                                                                            states = states,
                                                                            actions = actions,
                                                                            values = values,
                                                                            log_probs = log_probs,
                                                                            returns = returns,
                                                                            advantages = advantages,
                                                                            ):
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor loss
            surr_loss = ratio * adv
            clipped_surr_loss = (T.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv)

            entropy = dist.entropy().mean()
            # actor_loss = -T.min(surr_loss, clipped_surr_loss).mean()
            actor_loss = (-T.min(surr_loss, clipped_surr_loss).mean() - entropy * self.entropy_weight)

            # critic loss
            value = self.critic(state)
            critic_loss = (return_ - value).pow(2).mean()

            # loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            self.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic.optimizer.step()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []

    def save_models(self):
        print('------ save models ------')
        self.actor.save_models()
        self.critic.save_models()

    def load_models(self):
        print('------ load models ------')
        self.actor.load_models()
        self.critic.load_models()
