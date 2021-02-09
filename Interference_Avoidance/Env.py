import numpy as np
import math
import random

class Env():
    def __init__(self,):
        self.P_s = 5
        self.h_s = 0.8
        self.PSD_th = 2
        self.SINR_th = 2
        self.P_J_j = 8
        self.h_J_j = 0.7
        self.Sigma = 1
        self.Signal = self.P_s * self.h_s

        self.Ac = [1,2,3,4,5,6]
        self.As = []

    def get_P_I_i(self,):
        P_I_i = np.random.randint(3,7,1)
        return P_I_i

    def get_h_I_i(self,):
        a = np.random.uniform(0.4,0.9)
        h_I_i = round(a,2)
        return h_I_i

    def sinr_function(self, Signal, psd_input,):
        sinr_ = Signal / psd_input
        return sinr_

    def Ph_Ii(self, P_I_i, h_I_i,):
        return P_I_i * h_I_i

    def psd_function(self, Sigma, Ph_Ii_1, Ph_Ii_2,):
        psd_ = Sigma + Ph_Ii_1 + Ph_Ii_2
        return psd_

    def g_func(self,SINR_input):
        if SINR_input > self.SINR_th:
            Lambda = 10
        else:
            Lambda = -10
        return Lambda

    def f_func(self,PSD_input):
        if PSD_input > self.PSD_th:
            Lambda = -10
        else:
            Lambda = 10
        return Lambda

    def indication_matrix(self,):
        pass

    def reset(self,):
        pass

    def step(self,):
        pass