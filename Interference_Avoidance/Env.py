import numpy as np
import math
import random

class Env():
    def __init__(self):
        self.Ps = 5
        self.hs = 0.8
        self.PSD_th = 2
        self.SINR_th = 2
        self.Pj = 8
        self.hj = 0.7
        self.Sigma = 1
        self.Signal = self.P_s * self.h_s
        self.first_state = None

        self.Ac_spaces = [1,2,3,4,5,6]   # N = 6
        self.Ac = np.zeros([6,1])
        self.As_spaces = [1,2]           # Ns = 2
        self.As = np.zeros([2,1])

    def get_P_I_i(self):
        P_I_i = np.random.randint(3,7,1)
        return P_I_i

    def get_h_I_i(self):
        h_I_i = np.around(np.random.uniform(0.4,0.9),2)
        return h_I_i

    def sinr_func(self, Signal, noise_input):
        sinr = Signal / psd_input
        return sinr

    def Ph_Ii(self, P_I_i, h_I_i):
        return np.around(P_I_i * h_I_i ,2 )

    def psd_func(self, Sigma, Ph_Ii):
        psd = Sigma + Ph_Ii_1
        return psd

    def noise(self, Sigma, Ph_Ii_1, Ph_Ii_2):
        noise = Sigma + Ph_Ii_1 + Ph_Ii_2
        return noise

    def g_func(self,SINR_input):
        if SINR_input > self.SINR_th:
            Lambda = 10
            success = True
        else:
            Lambda = -10
            success = False
        return Lambda, success

    def f_func(self,PSD_input):
        if PSD_input > self.PSD_th:
            Lambda = -10
            availability = False
        else:
            Lambda = 10
            availability = True
        return Lambda, availability

    def reset(self):
        pi1 = get_P_I_i()
        pi2 = get_P_I_i()

        hi1 = get_h_I_i()
        hi2 = get_h_I_i()

        ph1 = Ph_Ii(pi1, hi1)
        ph2 = Ph_Ii(pi2, hi2)

        v1 = psd_func(self.Sigma, ph1)
        v2 = psd_func(self.Sigma, ph2)

        a = np.zeros([2,1])
        a1, _ = f_func(v1)
        a2, _ = f_func(v2)
        a[0][0] = a1
        a[1][0] = a2

        select_as = np.where(a == 10)[0]

        noise = noise(self.Sigma, ph1, ph2)
        sinr = sinr_func(self.Signal, noise)

        b , _ = g_func(sinr)

        def indication_matrix(self, g, f1, f2):
            I = np.zeros([3,2]) # shape (3,2)

            I[1][0] = 1
            I[2][0] = 2

            I[0][1] = g
            I[1][1] = f1
            I[2][1] = f2

        i = indication_matrix(b, a1, a2)

        s = np.stack((i, i, i), axis=0) # (3 , 3 , 2 )
        self.first_state = s

        return s

    def step(self,action):
        passpi1 = get_P_I_i()
        pi2 = get_P_I_i()

        hi1 = get_h_I_i()
        hi2 = get_h_I_i()

        ph1 = Ph_Ii(pi1, hi1)
        ph2 = Ph_Ii(pi2, hi2)

        v1 = psd_func(self.Sigma, ph1)
        v2 = psd_func(self.Sigma, ph2)

        a = np.zeros([2,1])
        a1, _ = f_func(v1)
        a2, _ = f_func(v2)
        a[0][0] = a1
        a[1][0] = a2

        select_as = np.where(a == 10)[0]

        noise = noise(self.Sigma, ph1, ph2)
        sinr = sinr_func(self.Signal, noise)

        b , _ = g_func(sinr)

        def indication_matrix(self, g, f1, f2):
            I = np.zeros([3,2])      # shape (3,2)

            I[1][0] = 1
            I[2][0] = 2

            I[0][1] = g
            I[1][1] = f1
            I[2][1] = f2

        i = indication_matrix(b, a1, a2)
        next_state = np.append(i, self.first_state[:2, :, :], axis=0)    # (3 , 3 , 2 )

        reward = sinr

        return next_state, reward