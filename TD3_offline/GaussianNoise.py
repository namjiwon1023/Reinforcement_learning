import random
import numpy as np

class GaussianNoise:

    def __init__(
        self,
        action_dim,
        min_sigma = 1.0,
        max_sigma = 1.0,
        decay_period = 1000000,
    ):
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t = 0) -> float:

        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(0, sigma, size=self.action_dim)