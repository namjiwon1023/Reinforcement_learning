import threading
import numpy as np
import torch as T

class Buffer:
    def __init__(self, args):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        '''The information collected by each agent must be stored'''
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = T.empty([self.size, self.args.obs_shape[i]],  dtype=T.float32, device=self.args.device)
            self.buffer['u_%d' % i] = T.empty([self.size, self.args.action_shape[i]],  dtype=T.float32, device=self.args.device)
            self.buffer['r_%d' % i] = T.empty([self.size],  dtype=T.float32, device=self.args.device)
            self.buffer['o_next_%d' % i] = T.empty([self.size, self.args.obs_shape[i]],  dtype=T.float32, device=self.args.device)
            self.buffer['done_%d' % i] = T.empty([self.size],  dtype=T.float32, device=self.args.device)
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next, done):
        idxs = self._get_storage_idx(inc=1) # Stored in the form of transition, only one experience at a time
        '''The information collected by each agent must be stored'''
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = T.as_tensor(o[i], dtype=T.float32, device=self.args.device)
                self.buffer['u_%d' % i][idxs] = T.as_tensor(u[i], dtype=T.float32, device=self.args.device)
                self.buffer['r_%d' % i][idxs] = T.as_tensor(r[i], dtype=T.float32, device=self.args.device)
                self.buffer['o_next_%d' % i][idxs] = T.as_tensor(o_next[i], dtype=T.float32, device=self.args.device)
                self.buffer['done_%d' % i][idxs] = T.as_tensor(done[i], dtype=T.float32, device=self.args.device)

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx