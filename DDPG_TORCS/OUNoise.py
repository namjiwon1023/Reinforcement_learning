import numpy as np
import random

class OUNoise():

    def OU(self, x, mu = 0, theta = 0.15, sigma = 0.2):
        """
        Ornstein-Uhlenbeck process.
        ou = θ * (μ - x) + σ * w

        {x: action value,
        mu: μ, mean fo values,
        theta: θ, rate the variable reverts towards to the mean,
        sigma：σ, degree of volatility of the process,
        }

        Returns: OU value
        """
        return theta * (mu - x) + sigma * np.random.randn(1)