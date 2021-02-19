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

    # Get the value of PIi
    def get_Pi(self):
        Pi = np.random.randint(3,7,1)
        return Pi

    # Get the value of hIi
    def get_hi(self):
        hi = np.around(np.random.uniform(0.4,0.91),2)
        return hi

    # Get the value of SINR
    def get_sinr(self, signal, noise):
        sinr = signal / noise
        return sinr

    #  Get the value of PIi*hIi
    def get_phi(self, pi, hi):
        return np.around(pi*hi, 2)


    # Calculation PSD
    def get_psd(self, sigma, ph1, ph2, ph3=None, phj=None):
        phj, pi3, hi3, ph3, noise = 0, 0, 0, 0, 0
        if phj and ph3 is not None :
            psd = sigma + ph1 + ph2 + ph3 + self.phj
        elif ph3 is not None:
            psd = sigma + ph1 + ph2 + ph3
        elif ph3 and phj is None:
            psd = sigma + ph1 + ph2
        return psd


    def get_noise(self):
        pass


    #  generate channel size : (channel_dims, 1)
    def generate_channel(self):
        channel = np.zeros([self.channel_dims, 1], dtype=np.float)
        return channel


    # PSD Evaluation function
    def f_func(self, psd):
        if psd > self.psd_th:
            Lambda = -10
        else:
            Lambda = 10
        return Lambda


    # SINR Evaluation function
    def g_func(self, sinr):
        if sinr > self.sinr_th:
            Lambda = 10
        else:
            Lambda = -10
        return 10*Lambda


    def reset(self, input_ph3=False, input_phj=False):
        ac_space = [0,1,2,3,4,5]
        index = []
        # Add noise and disturbers to the channel
        channel = self.generate_channel()
        # Add noise(sigma)
        for n in range(len(channel)):
            channel[n] += 1
        # Add case1 : inter 2 signal, case2 : inter 3 signal, case3 : inter 3 jammer 1 signal
        if input_ph3 is True:                       # case2
            index = np.random.choice(len(channel), 3, replace=True)
        elif input_ph3 and input_phj is False:      # case1
            index = np.random.choice(len(channel), 2, replace=True)
        elif input_phj is True:                     # case3
            index = np.random.choice(len(channel), 3, replace=True)
            phj_index = np.random.choice(len(channel), 1, replace=True)
            channel[phj_index] += self.phj
        # PH = []
        psd = []
        for i in range(len(index)):
            pi = self.get_Pi()
            hi = self.get_hi()
            phi = self.get_phi(pi,hi)
            channel[index[i]] += phi
            # PH.append(phi)
            psd.append(channel[i])
        ac_t = np.random.choice(len(channel),1,replace=False)
        # ac_space.remove(ac_t)
        as_t = np.random.choice(len(ac_space),2,replace=False)
        as_t_copy = copy.deepcopy(as_t)          # deepcopy : The two are completely independent
        self.next_ac_range = np.append([as_t_copy], [ac_t])

        # psd_t = get_psd(sigma=self.sigma, ph1=PH[1], ph2=PH[2], ph3=None, phj=None)
        psd_t = psd[ac_t]
        sinr_t = self.get_sinr(self.signal, psd_t)

        Ic_1 = np.array([[ac_t, self.g_func(sinr_t)]])
        Ic_2 = np.array([[as_t[0], self.f_func(channel[as_t[0]])]])
        Ic_3 = np.array([[as_t[1], self.f_func(channel[as_t[1]])]])
        Ic_t = np.concatenate((Ic_1, Ic_2, Ic_3), axis=0)

        s_t = np.stack((Ic_t, Ic_t, Ic_t), axis=0)
        state = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])    # reshape : (batch, dim, w, h)
        self.first_state = state
        return state

    def step(self, action, input_ph3=False, input_phj=False):
        ac_space = [0,1,2,3,4,5]
        index = []
        # Add noise and disturbers to the channel
        channel = self.generate_channel()
        # Add noise(sigma)
        for n in range(len(channel)):
            channel[n] += 1
        # Add case1 : inter 2 signal, case2 : inter 3 signal, case3 : inter 3 jammer 1 signal
        if input_ph3 is True:                       # case2
            index = np.random.choice(len(channel), 3, replace=True)
        elif input_ph3 and input_phj is False:      # case1
            index = np.random.choice(len(channel), 2, replace=True)
        elif input_phj is True:                     # case3
            index = np.random.choice(len(channel), 3, replace=True)
            phj_index = np.random.choice(len(channel), 1, replace=True)
            channel[phj_index] += self.phj
        # PH = []
        psd = []
        for i in range(len(index)):
            pi = self.get_Pi()
            hi = self.get_hi()
            phi = self.get_phi(pi,hi)
            channel[index[i]] += phi
            # PH.append(phi)
            psd.append(channel[i])
        # ac_t1 = np.random.choice(self.next_ac_range, 1, replace=False)
        ac_t1 = action
        # ac_space.remove(ac_t1)
        as_t1 = np.random.choice(len(ac_space),2,replace=False)
        as_t1_copy = copy.deepcopy(as_t1)     # deepcopy : The two are completely independent
        self.next_ac_range = np.append([as_t1_copy], [ac_t1])

        # psd_t1 = get_psd(sigma=self.sigma, ph1=PH[1], ph2=PH[2], ph3=None, phj=None)
        psd_t1 = psd[ac_t1]
        sinr_t1 = self.get_sinr(self.signal, psd_t1)

        Ic_1 = np.array([[ac_t1, self.g_func(sinr_t1)]])
        Ic_2 = np.array([[as_t1[0], self.f_func(channel[as_t1[0]])]])
        Ic_3 = np.array([[as_t1[1], self.f_func(channel[as_t1[1]])]])
        x_t1 = np.concatenate((Ic_1, Ic_2, Ic_3), axis=0)

        s_t1 = x_t1.reshape(1, 1, x_t1.shape[0],x_t1.shape[1])
        next_state =  np.append(x_t1, self.first_state[:, :2, :, :], axis=1)

        reward = sinr_t1
        done = False
        if ac_t1 not in self.next_ac_range:
            done = True

        return next_state, reward, done
