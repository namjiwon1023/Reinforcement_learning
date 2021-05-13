import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random

from Networks.ActorNetwork import Actor
from Networks.CriticNetwork import Critic
from Utils.ReplayBuffer import ReplayBuffer
from Utils.utils import make_env
import gym
from gym.spaces.discrete import Discrete
import copy




class maddpg(object):
    def __init__(self, args):
        self.env_id = args.scenario_name
        self.env = make_env(self.env_id)
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.value_iter = args.value_iter
        self.policy_iter = args.policy_iter
        self.tau = args.tau
        self.episode_len = args.max_episode_len
        self.train_start = args.train_start
        self.model_path = args.checkpoint

        self.obs_dims = self.env.observation_space
        self.act_dims = self.env.action_space

        self.obs_total_dims = sum([self.env.observation_space[i].shape[0] for i in range(self.env.n)])
        self.act_total_dims = sum([self.env.action_space[i].n if isinstance(self.env.action_space[i], Discrete) else sum(self.env.action_space[i].high) + self.env.action_space[i].shape for i in range(self.env.n)])

        self.actors = [Actor(self.obs_dims[i].shape[0], self.act_dims[i].n if isinstance(self.env.action_space[i], Discrete) else sum(self.env.action_space[i].high) + self.env.action_space[i].shape) for i in range(self.env.n)]
        self.actor_targets = copy.deepcopy(self.actors)

        self.critics = [Critic(self.obs_total_dims, self.act_total_dims, 1) for i in range(self.env.n)]
        self.critic_targets = copy.deepcopy(self.critics)

        if self.model_path:
            for i in range(self.env.n):
                self.actors[i] = torch.load('./models/{}/actor{}.pkl'.format(self.env_id, i))
                self.critics[i] = torch.load('./models/{}/critic{}.pkl'.format(self.env_id, i))

        self.buffer = ReplayBuffer( self.obs_dims, self.act_dims, args.memory_size)
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

            observations_stack = np.vstack([np.hstack(observations[b]) for b in range(len(observations))])
            # print('observations_stack : ',observations_stack)
            # print('observations_stack shape : ',observations_stack.shape)
            total_observations = torch.FloatTensor(observations_stack).view(len(observations), -1)
            #print('total_observations : ',total_observations.shape)
            indiv_observations = [torch.FloatTensor(np.vstack([observations[b][n] for b in range(self.batch_size)])) for n in range(self.env.n)]
            # print('indiv_observations : ',indiv_observations[:][0].shape)
            # print('indiv_observations : ',indiv_observations[0])
            # print('indiv_observations shape: ',indiv_observations[0].shape)
            #print('indiv_observations : ',len(indiv_observations))
            actions_stack = np.vstack([np.hstack(actions[b]) for b in range(len(actions))])
            #print('actions_stack : ',actions_stack.shape)
            indiv_actions = [torch.FloatTensor(np.vstack([actions[b][n] for b in range(self.batch_size)])) for n in range(self.env.n)]
            #print('indiv_actions : ',len(indiv_actions))
            total_actions = torch.cat(indiv_actions, dim=1)
            #print('total_actions : ',total_actions.shape)
            rewards = torch.FloatTensor(rewards)
            #print('rewards : ',rewards.shape)
            indiv_rewards = [rewards[:, n] for n in range(self.env.n)]
            # print('indiv_rewards : ',indiv_rewards)
            # print('indiv_rewards resize: ',indiv_rewards[0])
            # print('indiv_rewards resize: ',indiv_rewards[0].view(self.batch_size, -1))
            # print('indiv_rewards resize: ',indiv_rewards[0].unsqueeze(1))
            next_observations_stack = np.vstack([np.hstack(next_observations[b]) for b in range(len(next_observations))])
            total_next_observations = torch.FloatTensor(next_observations_stack).view(len(next_observations), -1)
            indiv_next_observations = [torch.FloatTensor(np.vstack([next_observations[b][n] for b in range(self.batch_size)])) for n in range(self.env.n)]
            dones = torch.FloatTensor(dones)
            #print('dones : ',dones.shape)
            indiv_dones = [dones[:, n] for n in range(self.env.n)]
            #print('indiv_dones : ',len(indiv_dones))

            for _ in range(self.value_iter):
                target_next_actions = torch.cat([self.target_policy_nets[n].forward(indiv_next_observations[n])[0] for n in range(self.env.n)], dim=1)

                target_next_value = self.target_value_nets[i].forward(total_next_observations, target_next_actions)
                q_target = indiv_rewards[i].unsqueeze(1) + self.gamma * (1 - indiv_dones[i].unsqueeze(1)) * target_next_value
                q_target = q_target.detach()
                q = self.value_nets[i].forward(total_observations, total_actions)
                value_loss = (q - q_target).pow(2).mean()

                self.value_optimizers[i].zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 0.5)
                self.value_optimizers[i].step()

            for _ in range(self.policy_iter):
                prob, entropy, origin_output = self.policy_nets[i].forward(indiv_observations[i])
                #action_idx = torch.LongTensor(actions)[:, i].unsqueeze(1)
                #action_prob = prob.gather(dim=1, index=action_idx)
                #policy_loss = -self.value_nets[i].forward(total_observations, total_actions).detach() * action_prob.log() - self.entropy_weight * entropy

                new_action = copy.deepcopy(indiv_actions)
                new_action[i] = prob
                new_actions = torch.cat(new_action, dim=1)
                policy_loss = - self.value_nets[i].forward(total_observations, new_actions) - self.entropy_weight * entropy

                pse_loss = torch.mean(torch.pow(origin_output, 2))
                policy_loss = policy_loss.mean()

                total_loss = 1e-3 * pse_loss + policy_loss

                self.policy_optimizers[i].zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_nets[i].parameters(), 0.5)
                self.policy_optimizers[i].step()

        self.soft_update()
        #self.buffer.clear()

    def run(self):
        max_reward = -np.inf
        save_flag = False
        weight_reward = [None for i in range(self.env.n)]
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = [0 for i in range(self.env.n)]
            if self.render:
                self.env.render()
            while True:
                actions = []
                for n in range(self.env.n):
                    action = self.policy_nets[n].act(torch.FloatTensor(np.expand_dims(obs[n], 0)))
                    actions.append(action)
                # * The action is real-value, not one-hot.
                next_obs, reward, done, info = self.env.step(actions)
                if self.render:
                    self.env.render()
                self.buffer.store(obs, actions, reward, next_obs, done)
                self.count += 1
                total_reward = [total_reward[i] + reward[i] for i in range(self.env.n)]
                obs = next_obs
                if self.count % self.train_freq == 0 and len(self.buffer) > self.batch_size and self.count > self.start_count:
                    self.train_count += 1
                    self.train()
                if any(done) or self.count % self.episode_len == 0:
                    for n in range(self.env.n):
                        if weight_reward[n] is not None:
                            weight_reward[n] = weight_reward[n] * 0.99 + total_reward[n] * 0.01
                        else:
                            weight_reward[n] = total_reward[n]
                    if max_reward < sum(weight_reward) and self.count > self.start_count:
                        max_reward = sum(weight_reward)
                        save_flag = True
                    if save_flag and self.count > self.start_count:
                        [torch.save(self.policy_nets[i], './models/{}/policy_model{}.pkl'.format(self.env_id, i)) for i in range(self.env.n)]
                        [torch.save(self.value_nets[i], './models/{}/value_model{}.pkl'.format(self.env_id, i)) for i in range(self.env.n)]
                    save_flag = False
                    print(('episode: {}\treward: '+'{:.2f}\t' * self.env.n).format(i + 1, *weight_reward))
                    break

    def eval(self):
        self.count = 0
        for i in range(self.env.n):
            self.policy_nets[i] = torch.load('./models/{}/policy_model{}.pkl'.format(self.env_id, i))
        while True:
            obs = self.env.reset()
            total_reward = [0 for i in range(self.env.n)]
            if self.render:
                self.env.render()
            while True:
                time.sleep(0.05)
                actions = []
                for n in range(self.env.n):
                    action = self.policy_nets[n].act(torch.FloatTensor(np.expand_dims(obs[n], 0)))
                    actions.append(action)
                next_obs, reward, done, info = self.env.step(actions)
                if self.render:
                    self.env.render()
                total_reward = [total_reward[i] + reward[i] for i in range(self.env.n)]
                obs = next_obs
                self.count += 1
                if any(done) or self.count % self.episode_len == 0:
                    print('episode: {}\treward: {}'.format(i + 1, total_reward))
                    break


