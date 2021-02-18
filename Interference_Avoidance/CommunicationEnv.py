import numpy as np
import math
import random

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

    def get_Pi(self):
        Pi = np.random.randint(3,7,1)
        return Pi

    def get_hi(self):
        hi = np.around(np.random.uniform(0.4,0.91),2)
        return hi

    def get_sinr(self, signal, noise):
        sinr = signal / noise
        return sinr

    def get_phi(self, pi, hi):
        return np.around(pi*hi, 2)

    def get_psd(self, sigma, ph1, ph2, input_ph3=False, input_phj=False):
        phj, pi3, hi3, ph3, noise = 0, 0, 0, 0, 0
        if input_phj and input_ph3 is True:
            phj = self.pj*self.hj
            pi3 = get_Pi()
            hi3 = get_hi()
            ph3 = pi3*hi3
            psd = sigma + ph1 + ph2 + ph3 + phj
        elif input_ph3 is True:
            pi3 = get_Pi()
            hi3 = get_hi()
            ph3 = pi3*hi3
            psd = sigma + ph1 + ph2 + ph3
        elif input_ph3 and input_phj is False:
            psd = sigma + ph1 + ph2
        return psd

    def get_noise(self):
        pass

    def generate_channel(self):
        channel = np.zeros([self.channel_dims, 1], dtype=np.float)
        return channel

    def f_func(self, psd):
        if psd > self.psd_th:
            Lambda = -10
        else:
            Lambda = 10
        return Lambda

    def g_func(self, sinr):
        if sinr > self.sinr_th:
            Lambda = 10
        else:
            Lambda = -10
        return 10*Lambda

    def reset(self, input_ph3=False, input_phj=False):
        # Add noise and disturbers to the channel
        channel = generate_channel()
        # Add noise(sigma)
        for n in range(len(channel)):
            channel[n] += 1
        # Add case1 : inter 2 signal, case2 : inter 3 signal, case3 : inter 3 jammer 1 signal
        if input_ph3 is True:
            index = np.random.choice(len(channel), 3, replace=True)
        elif input_ph3 and input_phj is False:
            index = np.random.choice(len(channel), 2, replace=True)
        elif input_phj is True:
            index = np.random.choice(len(channel), 3, replace=True)
            phj_index = np.random.choice(len(channel), 1, replace=True)
            channel[phj_index] += self.phj

        for i in range(len(index)):
            pi = get_Pi()
            hi = get_hi()
            phi = get_phi(pi,hi)
            channel[index[i]] += phi


    def step(self, action):
        pass

