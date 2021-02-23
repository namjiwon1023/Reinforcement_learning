import numpy as np
import math
import random
import copy

class CommunicationEnv:
    def __init__(self):
        self.ps = 5
        self.hs = 0.8
        self.psd_th = 2
        self.sinr_th = 2
        self.pj = 8
        self.hj = 0.7
        self.phj = self.pj*self.hj
        self.sigma = 1
        self.signal = self.ps*self.hs
        self.channel_dims = 6
        self.select_as_dims = 2
        self.next_ac_range = []
        self.first_state = None
    
    def generate_channel(self):
        channel = np.zeros([self.channel_dims, 1], dtype=np.float)
        return channel

    def get_Pi(self):
        Pi = np.random.randint(3,7,1)
        return Pi

    # Get the value of hIi
    def get_hi(self):
        hi = np.around(np.random.uniform(0.4,0.91), 2)
        return hi

    # Get the value of SINR
    def get_sinr(self, signal, noise):
        sinr = signal / noise
        return sinr

    #  Get the value of PIi*hIi
    def get_phi(self, pi, hi):
        return np.around(pi*hi, 2)


    def reset(self):
        ac_space = [0,1,2,3,4,5]
        index_t = None
        channel_t = None
        ac_t = None
        as_t = None
        as_t_copy = None
        psd_t = None
        sinr_t = None
        Ic_1 = None
        Ic_2 = None
        Ic_3 = None
        Ic_t = None
        s_t = None
        state = None
        self.first_state = None
        self.next_ac_range = []

        channel_t = self.generate_channel()

        for n in range(len(channel_t)):
            channel_t[n] += 1

        index_t = np.random.choice(len(channel_t), 4, replace=False)

        for i in range(len(index_t)):
            pi = self.get_Pi()
            hi = self.get_hi()
            phi = self.get_phi(pi,hi)
            channel_t[index_t[i]] += phi

        ac_t = np.random.choice(len(channel_t),1,replace=False)

        as_t = np.random.choice(len(ac_space),2,replace=False)
        as_t_copy = copy.deepcopy(as_t)        

        psd_t = channel_t[ac_t[0]][0]

        sinr_t = self.get_sinr(self.signal, psd_t)
        print('SINR : ', sinr_t)

if __name__ == "__main__":
    env = CommunicationEnv()
    for i in range(10):
        env.reset()