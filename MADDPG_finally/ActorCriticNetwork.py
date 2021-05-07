# -*- coding:utf8 -*-
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        '''The observations and actions of each agent are different, so use agent_id to distinguish'''
        self.actor = nn.Sequential(
                                nn.Linear(args.obs_shape[agent_id], args.n_hiddens),
                                nn.ReLU(),
                                nn.Linear(args.n_hiddens, args.n_hiddens),
                                nn.ReLU(),
                                nn.Linear(args.n_hiddens, args.n_hiddens),
                                nn.ReLU(),
                                nn.Linear(args.n_hiddens, args.action_shape[agent_id])
                                )
        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)
        self.to(args.device)

    def forward(self, x):
        x = self.actor(x)
        actions = self.max_action * T.tanh(x)

        return actions

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        '''MADDPG CriticNetwork Evaluate the state and actions of all agents'''
        self.critic = nn.Sequential(
                                nn.Linear(sum(args.obs_shape) + sum(args.action_shape), args.n_hiddens),
                                nn.ReLU(),
                                nn.Linear(args.n_hiddens, args.n_hiddens),
                                nn.ReLU(),
                                nn.Linear(args.n_hiddens, args.n_hiddens),
                                nn.ReLU(),
                                nn.Linear(args.n_hiddens, 1)
                                )
        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_lr)
        self.to(args.device)

    def forward(self, state, action):
        '''All states need to be spliced'''
        state = T.cat(state, dim=1)
        '''Every action must become within the range of [-1, 1]'''
        for i in range(len(action)):
            action[i] /= self.max_action
        '''All actions also need to be spliced'''
        action = T.cat(action, dim=1)
        '''DDPG algorithm CriticNetwork input data : cat([state, action], dim=-1)'''
        cat = T.cat([state, action], dim=1)
        value = self.critic(cat)

        return value
