import numpy as np
import torch as T
import random
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque
import gym
import matplotlib.pyplot as plt

from ActorCriticNetwork import ActorNetwork, CriticNetwork

from PPO import PPOAgent

if __name__ == '__main__':
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True

    seed = 777
    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    params = {
                'gamma' : 0.9,
                'lr_actor' : 1e-3,
                'lr_critic': 5e-3,
                'tau' : 0.8,
                'batch_size' : 64,
                'epoch' : 64,
                'epsilon' : 0.2,
                'entropy_weight' : 0.005,
                'test_mode' : False,
                'total_step' : 1,
                'rollout_len' : 2048,
}

    agent = PPOAgent(**params)
    agent.env.seed(seed)
    n_games = int(1e6)
    figure_file = '/home/nam/Reinforcement_learning/PPO_GAE_NEW/Pendulum.png'
    best_score = agent.env.reward_range[0]
    scores = []
    # N = 20
    learn_iters = 0
    avg_score = 0


    plt.ion()
    plt.figure(figsize=(10, 10))

    state = agent.env.reset()
    state = np.expand_dims(state, axis=0)
    score = 0
    np.savetxt("./Total_scores.txt",scores, delimiter=",")

    while agent.total_step <= n_games + 1:
        for _ in range(agent.rollout_len):
            agent.total_step += 1
            agent.env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
            reward = np.reshape(reward, (1, -1)).astype(np.float64)
            done = np.reshape(done, (1, -1))
            if not agent.test_mode:
                agent.rewards.append(T.FloatTensor(reward).to(agent.actor.device))
                agent.masks.append(T.FloatTensor(1- done).to(agent.actor.device))

            state = next_state
            score += reward[0][0]

            if done[0][0]:
                state = agent.env.reset()
                state = np.expand_dims(state, axis=0)
                scores.append(score)
                score = 0

        agent.learn(next_state)
        learn_iters += 1

        avg_score = np.mean(scores[-10:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode',agent.total_step,'score %.1f' %score,'avg score %.1f' % avg_score, 'learning_step',learn_iters)

        z = [c+1 for c in range(len(scores))]
        running_avg = np.zeros(len(scores))
        for e in range(len(running_avg)):
            running_avg[e] = np.mean(scores[max(0, e-10):(e+1)])
        plt.cla()
        plt.title("Total_scores")
        plt.grid(True)
        plt.xlabel("Episode_Reward")
        plt.ylabel("Total reward")
        plt.plot(scores, "r-", linewidth=1.5, label="PPO_Episode_Reward")
        plt.plot(z, running_avg, "b-", linewidth=1.5, label="PPO_Avg_Reward")
        plt.legend(loc="best", shadow=True)
        plt.pause(0.1)
        # plt.ioff()
        plt.show()


    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(x, scores, figure_file)
    agent.env.close()