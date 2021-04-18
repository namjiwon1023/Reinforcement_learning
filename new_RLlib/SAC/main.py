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
                'soft_update_time' : 1,
                'memory_size' : int(1e6),
                'policy_update_step' : int(2**10),
                'repeat_times' : int(2 ** 0),
                'batch_size' : 256,
                'total_step' : 0,
                'train_start_step' : 10000,
                'random_action_step' : 10000,
                'test_mode' : False,
                'use_cuda' : True,
                'learn_iter' : 0,
                'render' : False,
}
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    random_seed(123)
    agent = SACAgent(**params)

    dirPath = '/home/nam/Reinforcement_learning/new_RLlib/SAC/checkpoint/'
    sac_actor_parameter = dirPath + 'sac_actor'
    sac_actor_optimizer_parameter = dirPath + 'sac_actor_optimizer'
    sac_critic_parameter = dirPath + 'sac_critic'
    sac_critic_optimizer_parameter = dirPath + 'sac_critic_optimizer'
    alpha_optimizer_parameter = dirPath + 'alpha_optimizer'

    if os.path.exists(sac_actor_parameter) and os.path.exists(sac_actor_optimizer_parameter) and os.path.exists(sac_critic_parameter) and os.path.exists(sac_critic_optimizer_parameter) and os.path.exists(alpha_optimizer_parameter):
        agent.load_models()
    else:
        print('------ No parameters available! ------')

    i_episode = int(1e6)
    best_score = agent.env.reward_range[0]
    scores = []

    avg_score = 0
    n_steps = 0


    plt.ion()
    plt.figure(figsize=(10, 5))


    for i in range(1, i_episode + 1):
        state = agent.env.reset()
        episode_steps = 0

        done = False
        score = 0

        np.savetxt("./Total_scores.txt",scores, delimiter=",")

        while (not done):
            if agent.render is True:
                agent.env.render()
            agent.total_step += 1
            episode_steps += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            real_done = False if episode_steps >= agent.env.spec.max_episode_steps else done
            mask = 0.0 if real_done else agent.GAMMA
            score += reward
            agent.transition += [reward, next_state, mask]
            agent.memory.store(*agent.transition)
            state = next_state
            n_steps += 1

        if (len(agent.memory) >= agent.batch_size and agent.total_step > agent.train_start_step):
            agent.learn()

        scores.append(score)
        avg_score = np.mean(scores[-10:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('Episode',i,' | Episode_score %.1f' %score,' | Avg score %.1f' % avg_score, ' | Time_steps',n_steps, ' | learning_step',agent.learn_iter)
        if i % 10 == 0:
            _plot(scores)