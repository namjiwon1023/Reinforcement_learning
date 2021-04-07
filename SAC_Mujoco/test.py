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

from SAC import SACAgent
from utils import random_seed, _plot


if __name__ == '__main__':
    params = {
                'GAMMA' : 0.99,
                'learning_rate' : 3e-4,
                'tau' : 5e-3,
                'update_time' : 1,
                'memory_size' : int(1e6),
                'batch_size' : 256,
                'total_episode' : 10001,
                'train_start' : 10000,
                'test_mode' : True,
}
    random_seed(123)
    agent = SACAgent(**params)

    sac_actor_parameter = '/home/nam/Reinforcement_learning/SAC_Mujoco/Walker2d-v2/sac_actor'
    sac_actor_optimizer_parameter = '/home/nam/Reinforcement_learning/SAC_Mujoco/Walker2d-v2/sac_actor_optimizer'
    sac_critic_parameter = '/home/nam/Reinforcement_learning/SAC_Mujoco/Walker2d-v2/sac_critic'
    sac_critic_optimizer_parameter = '/home/nam/Reinforcement_learning/SAC_Mujoco/Walker2d-v2/sac_critic_optimizer'
    alpha_optimizer_parameter = '/home/nam/Reinforcement_learning/SAC_Mujoco/Walker2d-v2/alpha_optimizer'

    if os.path.exists(sac_actor_parameter) and os.path.exists(sac_actor_optimizer_parameter) and os.path.exists(sac_critic_parameter) and os.path.exists(sac_critic_optimizer_parameter) and os.path.exists(alpha_optimizer_parameter):
        agent.load_models()
    else:
        print('------ No parameters available! ------')

    i_episode = int(1e6)

    for i in range(1, i_episode + 1):
        state = agent.env.reset()
        done = False

        while (not done):
            agent.env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            state = next_state