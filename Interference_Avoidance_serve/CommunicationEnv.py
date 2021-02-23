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
        self.second_state = None
        self.n_steps = 0

    # Get the value of PIi
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


    # def reset(self, input_ph3=False, input_phj=False):
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
        # Add noise and disturbers to the channel
        channel_t = self.generate_channel()
        # Add noise(sigma)
        for n in range(len(channel_t)):
            channel_t[n] += 1
        # Add case1 : inter 2 signal, case2 : inter 3 signal, case3 : inter 3 jammer 1 signal
        # if input_ph3 is True:                       # case2
        #     index = np.random.choice(len(channel_t), 3, replace=True)
        # elif input_ph3 and input_phj is False:      # case1
        #     index = np.random.choice(len(channel_t), 2, replace=True)
        # elif input_phj is True:                     # case3
        #     index = np.random.choice(len(channel_t), 3, replace=True)
        #     phj_index = np.random.choice(len(channel_t), 1, replace=True)
        #     channel_t[phj_index] += self.phj

        # index_t = np.random.choice(len(channel_t), 2, replace=False)
        index_t = [1, 3]
        # PH = []
        # psd = []
        for i in range(len(index_t)):
            pi = self.get_Pi()
            hi = self.get_hi()
            phi = self.get_phi(pi,hi)
            channel_t[index_t[i]] += phi
            # PH.append(phi)
            # psd.append(channel[i])
        # ac_t = np.random.choice(len(channel_t),1,replace=False)
        ac_t = [0]
        # ac_space.remove(ac_t)
        as_t = np.random.choice(len(ac_space),2,replace=False)
        as_t_copy = copy.deepcopy(as_t)          # deepcopy : The two are completely independent
        # self.next_ac_range = np.append([as_t_copy], [ac_t])

        # psd_t = get_psd(sigma=self.sigma, ph1=PH[1], ph2=PH[2], ph3=None, phj=None)
        psd_t = channel_t[ac_t[0]][0]

        sinr_t = self.get_sinr(self.signal, psd_t)

        Ic_1 = np.array([[ac_t[0], self.g_func(sinr_t)]])
        Ic_2 = np.array([[as_t[0], self.f_func(channel_t[as_t[0]][0])]])
        Ic_3 = np.array([[as_t[1], self.f_func(channel_t[as_t[1]][0])]])

        Ic_t = np.concatenate((Ic_1, Ic_2, Ic_3), axis=0)

        s_t = np.stack((Ic_t, Ic_t, Ic_t), axis=0)
        state = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])    # reshape : (batch, dim, w, h)
        self.first_state = state
        self.next_ac_range = np.append([as_t_copy], [ac_t])
        self.n_steps += 1
        return state

    # def step(self, action, input_ph3=False, input_phj=False):
    def step(self, action):
        ac_space = [0,1,2,3,4,5]
        index_t1 = None
        channel_t1 = None
        done = None
        ac_t1 = None
        as_t1 = None
        as_t1_copy = None
        psd_t1 = None
        sinr_t1 = None
        Ic_1 = None
        Ic_2 = None
        Ic_3 = None
        Ic_t1 = None
        s_t1 = None
        next_state = None
        # Add noise and disturbers to the channel
        channel_t1 = self.generate_channel()
        # Add noise(sigma)
        for n in range(len(channel_t1)):
            channel_t1[n] += 1
        # Add case1 : inter 2 signal, case2 : inter 3 signal, case3 : inter 3 jammer 1 signal
        # if input_ph3 is True:                       # case2
        #     index = np.random.choice(len(channel_t1), 3, replace=True)
        # elif input_ph3 and input_phj is False:      # case1
        #     index = np.random.choice(len(channel_t1), 2, replace=True)
        # elif input_phj is True:                     # case3
        #     index = np.random.choice(len(channel_t1), 3, replace=True)
        #     phj_index = np.random.choice(len(channel_t1), 1, replace=True)
        #     channel_t1[phj_index] += self.phj
        #index_t1 = np.random.choice(len(channel_t1), 2, replace=False)
        index_t1 = [1, 3]
        # PH = []
        # psd = []
        for i in range(len(index_t1)):
            pi = self.get_Pi()
            hi = self.get_hi()
            phi = self.get_phi(pi,hi)
            channel_t1[index_t1[i]] += phi
            # PH.append(phi)
            # psd.append(channel[i])
        # ac_t1 = np.random.choice(self.next_ac_range, 1, replace=False)
        ac_t1 = action
        # ac_space.remove(ac_t1)
        as_t1 = np.random.choice(len(ac_space),2,replace=False)
        as_t1_copy = copy.deepcopy(as_t1)     # deepcopy : The two are completely independent
        # self.next_ac_range = np.append([as_t1_copy], [ac_t1])

        # psd_t1 = get_psd(sigma=self.sigma, ph1=PH[1], ph2=PH[2], ph3=None, phj=None)
        psd_t1 = channel_t1[ac_t1][0]
        sinr_t1 = self.get_sinr(self.signal, psd_t1)

        Ic_1 = np.array([[ac_t1, self.g_func(sinr_t1)]])
        Ic_2 = np.array([[as_t1[0], self.f_func(channel_t1[as_t1[0]][0])]])
        Ic_3 = np.array([[as_t1[1], self.f_func(channel_t1[as_t1[1]][0])]])

        Ic_t1 = np.concatenate((Ic_1, Ic_2, Ic_3), axis=0)

        s_t1 = Ic_t1.reshape(1, 1, Ic_t1.shape[0], Ic_t1.shape[1])

        if self.n_steps == 1:
            next_state = np.append(s_t1, self.first_state[:, :2, :, :], axis=1)
            # next_state = np.stack((s_t1, self.first_state[:, :1, :, :]), axis=1)
            # print('1 Next State : ',next_state)
            # print('1 next_state size : ',next_state.shape)
            self.first_state = self.first_state[:, :1, :, :]
            # print('1 first_state : ',self.first_state)
            # print('1 first_state size : ',self.first_state.shape)
        if self.n_steps == 2:
            next_state = np.append(s_t1, self.first_state, axis=1)
            next_state = np.append(next_state, self.second_state, axis=1)
            # next_state = np.append(s_t1, self.first_state[:, :2, :, :], axis=1)
            # next_state = np.stack((s_t1, self.second_state, self.first_state), axis=1)
            self.first_state = copy.deepcopy(self.second_state)
        elif self.n_steps >= 3:

            next_state = np.append(s_t1, self.first_state, axis=1)
            next_state = np.append(next_state, self.second_state, axis=1)
            # next_state = np.append(s_t1, [self.second_state, self.first_state], axis=1)
            # next_state = np.concatenate((s_t1, self.second_state, self.first_state), axis=0)
            # next_state = np.stack((s_t1, self.second_state, self.first_state), axis=1)
            self.first_state = copy.deepcopy(self.second_state)   
            # print('2 Next State : ',next_state) 
            # print('2 size : ',next_state.shape)
            # print('2 First State : ',self.first_state)
        # elif self.n_steps > 3:
            # next_state = np.stack((s_t1, self.second_state, self.first_state), axis=1)
            # self.first_state = copy.deepcopy(self.second_state)

        self.second_state = s_t1
        # next_state =  np.append(s_t1, self.first_state[:, :2, :, :], axis=1)

       
        if ac_t1 not in self.next_ac_range:
            reward = 0
            done = True
            self.n_steps = 0

        else:
            reward = sinr_t1
            done = False
            self.n_steps += 1

        self.next_ac_range = np.append([as_t1_copy], [ac_t1])
        

        return next_state, reward, done