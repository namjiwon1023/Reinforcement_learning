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


if __name__ == '__main__':
    params = {
                'GAMMA' : 0.99,
                'learning_rate' : 3e-4,
                'tau' : 0.005,
                'update_time' : 1,
                'memory_size' : int(1e6),
                'batch_size' : 256,
                'learn_step' : 0,
                'total_episode' : 0,
                'train_start' : 10000,
                'test_mode' : False,
}

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

    n_games = int(1e6)
    figure_file = '/home/nam/Reinforcement_learning/SAC_Mujoco/Walker2d-v3.png'
    best_score = agent.env.reward_range[0]
    scores = []
    # N = 20
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    plt.ion()
    plt.figure(figsize=(10, 5))


    for i in range(1, n_games + 1):
        state = agent.env.reset()
        # print('state : ', state)
        agent.total_episode = i

        done = False
        score = 0

        np.savetxt("./Total_scores.txt",scores, delimiter=",")

        while not done:
            agent.env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            # next_state = next_state.reshape(len(next_state))
            # print('next_state : ', next_state)
            n_steps += 1
            score += reward
            agent.transition += [reward, next_state, done]
            agent.memory.store(*agent.transition)
            if (len(agent.memory) >= agent.batch_size and agent.total_episode > agent.train_start):
                agent.learn()
                learn_iters += 1
            state = next_state
        scores.append(score)
        avg_score = np.mean(scores[-10:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode',i,'score %.1f' %score,'avg score %.1f' % avg_score, 'time_steps',n_steps, 'learning_step',learn_iters)

        z = [c+1 for c in range(len(scores))]
        running_avg = np.zeros(len(scores))
        for e in range(len(running_avg)):
            running_avg[e] = np.mean(scores[max(0, e-10):(e+1)])
        plt.cla()
        plt.title("Total_scores")
        plt.grid(True)
        plt.xlabel("Episode_Reward")
        plt.ylabel("Total reward")
        plt.plot(scores, "r-", linewidth=1.5, label="SAC_Episode_Reward")
        plt.plot(z, running_avg, "b-", linewidth=1.5, label="SAC_Avg_Reward")
        plt.legend(loc="best", shadow=True)
        plt.pause(0.1)
        plt.savefig('./SAC_TORCS.jpg')
        plt.show()
