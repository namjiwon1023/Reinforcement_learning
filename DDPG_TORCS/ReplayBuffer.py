import numpy as np

class ReplayBuffer:

    def __init__(self, obs_dim:int, memory_size : int, batch_size : int = 32):

        self.obs_buf = np.zeros([memory_size, obs_dim], dtype = np.float32)
        self.next_obs_buf = np.zeros([memory_size, obs_dim], dtype = np.float32)
        self.acts_buf = np.zeros([memory_size], dtype = np.float32)
        self.rews_buf = np.zeros([memory_size], dtype = np.float32)
        self.done_buf = np.zeros([memory_size], dtype = np.float32)

        self.max_size, self.batch_size = memory_size, batch_size
        self.ptr, self.size, = 0,0

    def store(self,
            obs:np.ndarray,
            act : np.ndarray,
            rew:float,
            next_obs:np.ndarray,
            done:bool ):

        self.obs_buf[self.ptr]=obs
        self.next_obs_buf[self.ptr]=next_obs
        self.acts_buf[self.ptr]=act
        self.rews_buf[self.ptr]=rew
        self.done_buf[self.ptr]=done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size +1, self.max_size)

    def sample_batch(self):

        index = np.random.choice(self.size, size = self.batch_size , replace=False)

        return dict(obs = self.obs_buf[index],
                    next_obs = self.next_obs_buf[index],
                    acts = self.acts_buf[index],
                    rews = self.rews_buf[index],
                    done = self.done_buf[index])

    def __len__(self):

        return self.size