import numpy as np
import torch as T

class ReplayBuffer:
    def __init__(self, memory_size, n_states, n_actions, use_cuda=False):

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.state = T.empty([memory_size, n_states], dtype=T.float32, device=self.device)
            self.next_state = T.empty([memory_size, n_states], dtype=T.float32, device=self.device)
            self.action = T.empty([memory_size, n_actions],dtype=T.float32, device=self.device)
            self.reward = T.empty([memory_size], dtype=T.float32, device=self.device)
            self.mask = T.empty([memory_size],dtype=T.float32, device=self.device)
        else:
            self.state = np.empty([memory_size, n_states], dtype=np.float32)
            self.next_state = np.empty([memory_size, n_states], dtype=np.float32)
            self.action = np.empty([memory_size, n_actions],dtype=np.float32)
            self.reward = np.empty([memory_size], dtype=np.float32)
            self.mask = np.empty([memory_size],dtype=np.float32)

        self.max_size = memory_size
        self.ptr, self.cur_len, = 0, 0
        self.n_states = n_states
        self.n_actions = n_actions
        self.count = 0

    def store(self, state, action, reward, next_state, mask):
        if self.use_cuda:
            self.state[self.ptr] = T.as_tensor(state, device=self.device)
            self.action[self.ptr] = T.as_tensor(action, device=self.device)
            self.reward[self.ptr] = T.as_tensor(reward, device=self.device)
            self.next_state[self.ptr] = T.as_tensor(next_state, device=self.device)
            self.mask[self.ptr] = T.as_tensor(mask, device=self.device)

        else:
            self.state[self.ptr] = state
            self.action[self.ptr] = action
            self.reward[self.ptr] = reward
            self.next_state[self.ptr] = next_state
            self.mask[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)
        self.count += 1

    def sample_batch(self, batch_size, fast_start=False):
        if fast_start is True:
            if self.count < batch_size:
                index = np.random.choice(self.cur_len, self.count)
            else:
                index = np.random.choice(self.cur_len, batch_size, replace = False)
        else:
            index = np.random.choice(self.cur_len, batch_size, replace = False)

        if self.use_cuda is True:
            return dict(state = T.as_tensor(self.state[index], dtype=T.float32, device=self.device),
                        action = T.as_tensor(self.action[index], dtype=T.float32, device=self.device),
                        reward = T.as_tensor(self.reward[index], dtype=T.float32, device=self.device).reshape(-1, 1),
                        next_state = T.as_tensor(self.next_state[index], dtype=T.float32, device=self.device),
                        mask = T.as_tensor(self.mask[index], dtype=T.float32, device=self.device).reshape(-1, 1),
                        )
        else:
            return dict(state = self.state[index],
                        action = self.action[index],
                        reward = self.reward[index],
                        next_state = self.next_state[index],
                        mask = self.mask[index],
                        )

    def clear(self):
        if self.use_cuda:
            self.state = T.empty([self.max_size, self.n_states], dtype=T.float32, device=self.device)
            self.next_state = T.empty([self.max_size, self.n_states], dtype=T.float32, device=self.device)
            self.action = T.empty([self.max_size, self.n_actions],dtype=T.float32, device=self.device)
            self.reward = T.empty([self.max_size], dtype=T.float32, device=self.device)
            self.mask = T.empty([self.max_size],dtype=T.float32, device=self.device)
        else:
            self.state = np.empty([self.max_size, self.n_states], dtype=np.float32)
            self.next_state = np.empty([self.max_size, self.n_states], dtype=np.float32)
            self.action = np.empty([self.max_size, self.n_actions],dtype=np.float32)
            self.reward = np.empty([self.max_size], dtype=np.float32)
            self.mask = np.empty([self.max_size], dtype=np.float32)

        self.ptr, self.cur_len, = 0, 0
        self.count = 0


    def store_for_BC_data(self, transitions):
        for t in transitions:
            self.store(*t)

    def __len__(self):
        return self.cur_len