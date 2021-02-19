import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os

from CriticNet import CriticNet
from ReplayBuffer import ReplayBuffer
from CommunicationEnv import CommunicationEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDQNAgent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.eval_net = CriticNet(self.input_dims, self.action_size).to(device)
        self.target_net = CriticNet(self.input_dims, self.action_size).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayBuffer(self.memory_size, self.batch_size)
        self.transition = list()

        self.optimizer = optim.SGD(self.eval_net.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()

        self.C_L = 0.
        self.Q_V = 0.
        self.chkpt_dir = '/home/nam/Reinforcement_learning/Interference_Avoidance'
        self.checkpoint_file = os.path.join(self.chkpt_dir, 'ddqn')

        if self.load_model :
            self.eval_net.load_state_dict(torch.load(self.checkpoint_file))

    def choose_action(self, state):

        s = torch.FloatTensor(state).to(device)

        if self.epsilon > np.random.random():
            choose_action = np.random.randint(0,self.action_size)
        else :
            choose_action = self.eval_net(s).to(device).argmax()
            choose_action = choose_action.detach().cpu().numpy()
        self.transition = [state, choose_action]
        return choose_action

    def target_net_update(self):

        if self.update_counter % self.update_time == 0 :
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.update_counter += 1


    def learn(self):

        self.target_net_update()
        samples = self.memory.sample_batch()

        state = torch.FloatTensor(samples['obs']).to(device)
        next_state = torch.FloatTensor(samples['next_obs']).to(device)
        action = torch.LongTensor(samples['act']).reshape(-1,1).to(device)
        reward = torch.FloatTensor(samples['rew']).reshape(-1,1).to(device)
        done = torch.FloatTensor(samples['done']).reshape(-1,1).to(device)

        curr_q = self.eval_net(state).gather(1, action)
        # print('curr_q : ',curr_q)
        self.Q_V = curr_q.detach().cpu().numpy()
        next_q = self.target_net(next_state).gather(1, self.eval_net(next_state).argmax(dim = 1, keepdim = True)).detach()
        # print('next_q : ',next_q)
        mask = 1 - done

        target_q = (reward + self.gamma * next_q * mask).to(device)
        # print('target_q : ',target_q)
        loss = self.loss(curr_q, target_q).to(device)
        self.C_L = loss.detach().cpu().numpy()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
