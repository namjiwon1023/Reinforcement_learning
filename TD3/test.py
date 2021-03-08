import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import gym
import random
import matplotlib.pyplot as plt

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from GaussianNoise import GaussianNoise
from TD3 import TD3Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    params = {
                'GAMMA' : 0.99,
                'tau' : 0.005,
                'exploration_noise' : 0.1,
                'policy_noise' : 0.2,
                'noise_clip' : 0.5,
                'actor_lr' : 3e-4,
                'critic_lr' : 1e-3,
                'update_time' : 2,
                'memory_size' : 100000,
                'batch_size' : 128,
                'learn_step' : 0,
                'total_episode' : 0,
                'train_start' : 1000,
                'test_mode' : True,
}
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True

    seed = 777
    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    agent = TD3Agent(**params)
    agent.load_models()
    n_games = int(10)


    for i in range(1, n_games + 1):
        state = agent.env.reset()
        done = False
        score = 0

        while not done:
            agent.env.render()
            action = agent.choose_action(state, agent.n_actions, test_mode=True)
            next_state, reward, done, _ = agent.env.step(action)
            score += reward
            state = next_state
        print('Score : ', score)
        agent.env.close()
