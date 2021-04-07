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
                'total_episode' : 0,
                'train_start' : 10000,
                'test_mode' : False,
}
    random_seed(123)
    agent = SACAgent(**params)

    sac_actor_parameter = '/home/nam/Reinforcement_learning/SAC_Mujoco/sac_actor'
    sac_actor_optimizer_parameter = '/home/nam/Reinforcement_learning/SAC_Mujoco/sac_actor_optimizer'
    sac_critic_parameter = '/home/nam/Reinforcement_learning/SAC_Mujoco/sac_critic'
    sac_critic_optimizer_parameter = '/home/nam/Reinforcement_learning/SAC_Mujoco/sac_critic_optimizer'
    alpha_optimizer_parameter = '/home/nam/Reinforcement_learning/SAC_Mujoco/alpha_optimizer'

    if os.path.exists(sac_actor_parameter) and os.path.exists(sac_actor_optimizer_parameter) and os.path.exists(sac_critic_parameter) and os.path.exists(sac_critic_optimizer_parameter) and os.path.exists(alpha_optimizer_parameter):
        agent.load_models()
    else:
        print('------ No parameters available! ------')

    i_episode = int(1e6)
    best_score = agent.env.reward_range[0]
    scores = []

    avg_score = 0
    learn_iter = 0
    n_steps = 0


    plt.ion()
    plt.figure(figsize=(10, 5))


    for i in range(1, i_episode + 1):
        state = agent.env.reset()

        agent.total_episode = i
        episode_steps = 0

        done = False
        score = 0


        np.savetxt("./Total_scores.txt",scores, delimiter=",")

        while (not done):
            agent.env.render()
            episode_steps += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            real_done = False if episode_steps >= agent.env.spec.max_episode_steps else done
            score += reward
            agent.transition += [reward, next_state, real_done]
            agent.memory.store(*agent.transition)
            if (len(agent.memory) >= agent.batch_size and agent.total_episode > agent.train_start):
                agent.learn()
                learn_iter += 1
            state = next_state
            n_steps += 1
        scores.append(score)
        avg_score = np.mean(scores[-10:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('Episode',i,' | Episode_score %.1f' %score,' | Avg score %.1f' % avg_score, ' | Time_steps',n_steps, ' | learning_step',learn_iter)
        if i % 100 == 0:
            _plot(scores)