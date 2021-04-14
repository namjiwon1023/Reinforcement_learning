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
        self.ptr, self.size, = 0, 0

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
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        index = np.random.choice(self.size, batch_size, replace = False)

        return dict(state = T.as_tensor(self.state[index], dtype=T.float32, device=self.device),
                    action = T.as_tensor(self.action[index], dtype=T.float32, device=self.device),
                    reward = T.as_tensor(self.reward[index], dtype=T.float32, device=self.device),
                    next_state = T.as_tensor(self.next_state[index], dtype=T.float32, device=self.device),
                    mask = T.as_tensor(self.mask[index], dtype=T.float32, device=self.device),
                    )

    def __len__(self):
        return self.size