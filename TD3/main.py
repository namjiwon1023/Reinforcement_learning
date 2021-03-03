import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import gym

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
                'train_start' : 1000
}

    agent = TD3Agent(**params)
    n_games = 50000
    figure_file = '/home/nam/Reinforcement_learning/TD3/Pendulum.png'
    best_score = agent.env.reward_range[0]
    scores = []
    # N = 20
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(1, n_games + 1):
        state = agent.env.reset()
        # print('state : ', state)
        agent.total_episode = i

        done = False
        score = 0
        while not done:
            agent.env.render()
            action = agent.choose_action(state, agent.n_actions)
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

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(x, scores, figure_file)