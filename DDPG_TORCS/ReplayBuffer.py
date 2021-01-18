import numpy as np

class ReplayBuffer:

    def __init__(self, obs_dim , memory_size , batch_size = 32, act_dim = 3):

        self.obs_buf = np.zeros([memory_size, obs_dim], dtype = np.float32)
        self.next_obs_buf = np.zeros([memory_size, obs_dim], dtype = np.float32)
        self.acts_buf = np.zeros([memory_size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([memory_size], dtype=np.float32)
        self.done_buf = np.zeros([memory_size], dtype=np.float32)

        self.max_size, self.batch_size = memory_size, batch_size
        self.ptr, self.size, = 0,0
        self.count = 0

    def store(self, obs, act , rew, next_obs, done):

        self.obs_buf[self.ptr]=obs
        self.next_obs_buf[self.ptr]=next_obs
        self.acts_buf[self.ptr]=act
        self.rews_buf[self.ptr]=rew
        self.done_buf[self.ptr]=done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size +1, self.max_size)
        self.count += 1

    def sample_batch(self):
        if self.count < self.batch_size:
            index = np.random.choice(self.size, self.count)
        else:
            index = np.random.choice(self.size, self.batch_size , replace=False)

        return dict(obs = self.obs_buf[index],
                    act = self.acts_buf[index],
                    rew = self.rews_buf[index],
                    next_obs = self.next_obs_buf[index],
                    done = self.done_buf[index])

    def __len__(self):

        return self.size