import numpy as np
import copy
import random

class OUNoise:

    def __init__(self,
                size = 1,
                mu = 0.0,
                theta = 0.15,
                sigma = 0.2,
        ):
        self.size = size
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
            )
        self.state = x + dx
        return self.state