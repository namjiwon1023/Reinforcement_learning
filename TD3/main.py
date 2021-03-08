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
                'test_mode' : False,
}
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True

    seed = 777
    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    agent = TD3Agent(**params)
    n_games = int(1e6)
    figure_file = '/home/nam/Reinforcement_learning/TD3/Pendulum.png'
    best_score = agent.env.reward_range[0]
    scores = []
    # N = 20
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    plt.ion()
    plt.figure()


    for i in range(1, n_games + 1):
        state = agent.env.reset()
        # print('state : ', state)
        agent.total_episode = i

        done = False
        score = 0

        np.savetxt("./Total_scores.txt",scores, delimiter=",")

        while not done:
            agent.env.render()
            action = agent.choose_action(state, agent.n_actions, agent.test_mode)
            next_state, reward, done, _ = agent.env.step(action)
            # next_state = next_state.reshape(len(next_state))
            # print('next_state : ', next_state)
            n_steps += 1
            score += reward
            agent.transition += [reward, next_state, done]
            agent.memory.store(*agent.transition)
            if (len(agent.memory) >= agent.batch_size and agent.total_episode > agent.train_start):
                # if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            state = next_state
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode',i,'score %.1f' %score,'avg score %.1f' % avg_score, 'time_steps',n_steps, 'learning_step',learn_iters)

        z = [c+1 for c in range(len(scores))]
        running_avg = np.zeros(len(scores))
        for e in range(len(running_avg)):
            running_avg[e] = np.mean(scores[max(0, e-100):(e+1)])
        plt.cla()
        plt.title("Total_scores")
        plt.grid(True)
        plt.xlabel("Episode_Reward")
        plt.ylabel("Total reward")
        plt.plot(scores, "r-", linewidth=2.0, label="TD3_Episode_Reward")
        plt.plot(z, running_avg, "b-", linewidth=2.0, label="TD3_Avg_Reward")
        plt.legend(loc="best", shadow=True)
        plt.pause(0.1)
        # plt.ioff()
        plt.show()


    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(x, scores, figure_file)