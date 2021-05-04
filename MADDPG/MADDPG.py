# -*- coding:utf8 -*-
import torch as T
import torch.optim as optim
import os
from ActorCriticNetwork import Actor, Critic
import numpy as np
import random
import copy
import pickle

class MADDPG:
    '''# Because the obs and act dimensions of different agents may be different, so the neural network is different, and agent_id is needed to distinguish'''
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # Create the Actor-Critic Network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)

        # build up the actor target network
        self.actor_target_network = copy.deepcopy(self.actor_network)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.actor_target_network.eval()
        for p in self.actor_target_network.parameters():
            p.requires_grad = False

        # build up the critic target network
        self.critic_target_network = copy.deepcopy(self.critic_network)
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.critic_target_network.eval()
        for q in self.critic_target_network.parameters():
            q.requires_grad = False

        # create the optimizer
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # load model
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(T.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(T.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor network: {}'.format(self.agent_id, self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic network: {}'.format(self.agent_id, self.model_path + '/critic_params.pkl'))

    def _soft_update_target_network(self):
        with T.no_grad():
            for t_p, l_p in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
                t_p.data.copy_(self.args.tau * l_p.data + (1 - self.args.tau) * t_p.data)

            for t_p, l_p in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
                t_p.data.copy_(self.args.tau * l_p.data + (1 - self.args.tau) * t_p.data)