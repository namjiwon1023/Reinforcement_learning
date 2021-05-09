# -*- coding:utf8 -*-
import numpy as np
import torch as T

def ReplayBuffer:
    def __init__(self, memory_size, n_states, n_actions, use_cuda=False, device):
        self.device = device
        self.use_cuda = use_cuda
        self.ptr, self.cur_len = 0, 0
        self.max_size = memory_size
        self.count = 0
        self.n_states = n_states
        self.n_actions = n_actions

        if self.use_cuda:
            self.states = T.empty([memory_size, n_states], dtype=T.float32, device=self.device)
            self.next_states = T.empty([memory_size, n_states], dtype=T.float32, device=self.device)
            self.actions = T.empty([memory_size, n_actions], dtype=T.float32, device=self.device)
            self.rewards = T.empty([memory_size], dtype=T.float32, device=self.device)
            self.masks = T.empty([memory_size], dtype=T.float32, device=self.device)
        else:
            self.states = np.empty([memory_size, n_states], dtype=np.float32)
            self.next_states = np.empty([memory_size, n_states], dtype=np.float32
            self.actions = np.empty([memory_size, n_actions], dtype=np.float32)
            self.rewards = np.empty([memory_size], dtype=np.float32)
            self.masks = np.empty([memory_size], dtype=np.float32)

    def store(self, state, action, reward, next_state, mask):
        if self.use_cuda:
            self.states[self.ptr] = T.as_tensor(state, device=self.device)
            self.next_states[self.ptr] = T.as_tensor(next_state, device=self.device)
            self.actions[self.ptr] = T.as_tensor(action, device=self.device)
            self.rewards[self.ptr] = T.as_tensor(reward, device=self.device)
            self.masks[self.ptr] = T.as_tensor(mask, device=self.device)

        else:
            self.states[self.ptr] = state
            self.next_states[self.ptr] = next_state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.masks[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)
        self.count += 1

    def sample_batch(self, batch_size, fast_start=False):
        if fast_start:
            if self.count < batch_size:
                index = np.random.choice(self.cur_len, self.count)
            else:
                index = np.random.choice(self.cur_len, batch_size, replace = False)
        else:
            index = np.random.choice(self.cur_len, batch_size, replace = False)

        if self.use_cuda:
            return dict(
                        state = T.as_tensor(self.states[index], dtype=T.float32, device=self.device),
                        next_state = T.as_tensor(self.next_states[index], dtype=T.float32, device=self.device),
                        action = T.as_tensor(self.actions[index], dtype=T.float32, device=self.device),
                        reward = T.as_tensor(self.rewards[index], dtype=T.float32, device=self.device),
                        mask = T.as_tensor(self.masks[index], dtype=T.float32, device=self.device)
                        )
        else:
            return dict(
                        state = self.states[index],
                        action = self.actions[index],
                        reward = self.rewards[index],
                        next_state = self.next_states[index],
                        mask = self.masks[index],
                        )

    def clear(self):
        if self.use_cuda:
            self.states = T.empty([self.max_size, self.n_states], dtype=T.float32, device=self.device)
            self.next_states = T.empty([self.max_size, self.n_states], dtype=T.float32, device=self.device)
            self.actions = T.empty([self.max_size, self.n_actions],dtype=T.float32, device=self.device)
            self.rewards = T.empty([self.max_size], dtype=T.float32, device=self.device)
            self.masks = T.empty([self.max_size],dtype=T.float32, device=self.device)
        else:
            self.states = np.empty([self.max_size, self.n_states], dtype=np.float32)
            self.next_states = np.empty([self.max_size, self.n_states], dtype=np.float32)
            self.actions = np.empty([self.max_size, self.n_actions],dtype=np.float32)
            self.rewards = np.empty([self.max_size], dtype=np.float32)
            self.masks = np.empty([self.max_size], dtype=np.float32)

        self.ptr, self.cur_len, = 0, 0
        self.count = 0


    def store_expert_data(self, transitions):
        for t in transitions:
            self.store(*t)

    def __len__(self):
        return self.cur_len
