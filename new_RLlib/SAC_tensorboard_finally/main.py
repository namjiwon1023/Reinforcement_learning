import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import gym
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

from SAC import SACAgent
from utils import random_seed


if __name__ == '__main__':
    writer = SummaryWriter()
    random_seed(123)
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
                'gradient_steps' : 1000,
                'target_update_interval' : 1,
                'learning_steps' : 0,
            }

    agent = SACAgent(**params)

    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

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

    i_episode = int(1e6)
    episode_max_step = int(1e4)
    max_steps = int(3e6)

    best_score = agent.env.reward_range[0]

    scores = []
    store_scores = []
    eval_rewards = []

    avg_score = 0
    n_updates = 0

    plt.ion()
    plt.figure(figsize=(20, 5))

    for i in range(1, i_episode + 1):
        state = agent.env.reset()
        cur_episode_steps = 0
        score = 0
        done = False

        if agent.total_step % agent.gradient_steps == 0 and agent.total_step > agent.train_start_step:
            Q1_loss, Q2_loss, Policy_loss, Alpha_loss = agent.learn()
            n_updates += 1

        if agent.total_step % agent.eval_steps == 0 and agent.total_step > agent.train_start_step:
            running_reward = np.mean(scores)
            eval_reward = agent.evaluate_agent(n_starts=10)
            writer.add_scalar('Loss/Q-Func1', Q1_loss, n_updates)
            writer.add_scalar('Loss/Q-Func2', Q2_loss, n_updates)
            writer.add_scalar('Loss/Policy', Policy_loss, n_updates)
            writer.add_scalar('Loss/Alpha', Alpha_loss, n_updates)
            writer.add_scalar('Reward/Train', running_reward, agent.total_step)
            writer.add_scalar('Reward/Test', eval_reward, agent.total_step)
            eval_rewards.append(eval_reward)
            scores = []
            print('| Episode : {} | Score : {} | Predict Score : {} | Avg score : {} |'.format(i, score, eval_reward, avg_score))

        for _ in range(1, episode_max_step + 1):
            if agent.render is True:
                agent.env.render()
            cur_episode_steps += 1
            agent.total_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            real_done = False if cur_episode_steps >= agent.env.spec.max_episode_steps else done
            mask = 0.0 if real_done else agent.GAMMA
            agent.transition += [reward, next_state, mask]
            agent.memory.store(*agent.transition)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)
        store_scores.append(score)
        avg_score = np.mean(scores[-10:])

        np.savetxt("./Episode_return.txt", store_scores, delimiter=",")
        np.savetxt("./Step_return.txt", eval_rewards, delimiter=",")

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        if agent.total_step >= max_steps:
            break

        print('Episode : {} | Score : {} | Avg score : {} | Time_Step : {} | Learning Step : {} | update number : {} |'.format(i, score, avg_score, agent.total_step, agent.learning_steps, n_updates))

    agent.env.close()