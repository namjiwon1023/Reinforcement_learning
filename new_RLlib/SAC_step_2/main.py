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
                'memory_size' : int(1e6),
                'batch_size' : int(2**8),
                'total_step' : 0,
                'train_start_step' : 10000,
                'render' : False,
                'eval_steps' : 1000,
                'gradient_steps' : 1,
                'target_update_interval' : 1,
}
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    random_seed(123)
    agent = SACAgent(**params)
    dirPath = os.getcwd() + '/checkpoint'
    sac_actor_parameter = dirPath + '/sac_actor'
    sac_actor_optimizer_parameter = dirPath + '/sac_actor_optimizer'
    sac_critic_parameter = dirPath + '/sac_critic'
    sac_critic_optimizer_parameter = dirPath + '/sac_critic_optimizer'
    alpha_optimizer_parameter = dirPath + '/alpha_optimizer'

    if os.path.exists(sac_actor_parameter) and os.path.exists(sac_actor_optimizer_parameter) \
        and os.path.exists(sac_critic_parameter) and os.path.exists(sac_critic_optimizer_parameter) \
        and os.path.exists(alpha_optimizer_parameter):
        agent.load_models()
    else:
        print('|------------------------------------|')
        print('|----- No parameters available! -----|')
        print('|------------------------------------|')

    i_episode = int(1e4)
    best_score = agent.env.reward_range[0]
    scores = []
    eval_rewards = []

    avg_score = 0
    n_steps = 0
    learning_steps = 0

    plt.ion()
    plt.figure(figsize=(20, 5))

    for i in range(1, i_episode + 1):
        state = agent.env.reset()
        episode_steps = 0
        score = 0
        done = False

        np.savetxt("./Episode_return.txt", scores, delimiter=",")
        np.savetxt("./Step_return.txt", eval_rewards, delimiter=",")

        while not done:
            if agent.render is True:
                agent.env.render()
            episode_steps += 1
            agent.total_step += 1
            n_steps += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            real_done = False if episode_steps >= agent.env.spec.max_episode_steps else done
            mask = 0.0 if real_done else agent.GAMMA
            agent.transition += [reward, next_state, mask]
            agent.memory.store(*agent.transition)
            state = next_state
            score += reward

        if agent.total_step >= agent.train_start_step and agent.total_step % agent.gradient_steps == 0:
            agent.learn()
            learning_steps += 1

        if agent.total_step >= agent.train_start_step and agent.total_step % agent.eval_steps == 0:
            eval_reward = agent.evaluate_agent(n_starts=10)
            eval_rewards.append(eval_reward)
            print('| Episode : {} | Score : {} | Predict Score : {} | Avg score : {} |'.format(i, score, eval_reward, avg_score))

        if agent.total_step > agent.train_start_step:
            scores.append(score)
            avg_score = np.mean(scores[-10:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        if i % 10 == 0:
            _plot(scores, eval_rewards)

        print('Episode : {} | Score : {} | Avg score : {} | Time_Step : {} | Learning Step : {} |'.format(i, score, avg_score, n_steps, learning_steps))

    agent.env.close()