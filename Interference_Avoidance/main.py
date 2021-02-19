import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os

from utils import plot_learning_curve

from CriticNet import CriticNet
from ReplayBuffer import ReplayBuffer
from CommunicationEnv import CommunicationEnv
from DDQN import DDQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    params = {
                'memory_size' : 2000,
                'batch_size' : 5,
                'input_dims' : 3,
                'action_size' : 6,
                'epsilon' : 0.1,
                'startTime' : 20,
                'update_time' : 10,
                'update_counter' : 0,
                'gamma' : 0.4,
                'lr' : 0.1,
                'load_model' : False,
                'load_episode' : 0,
        }

    agent = DDQNAgent(**params)

    chkpt_dir = '/home/nam/Reinforcement_learning/Interference_Avoidance'
    checkpoint_file = os.path.join(chkpt_dir, 'ddqn')
    figure_file = '/home/nam/Reinforcement_learning/Interference_Avoidance/plots/Interference_Avoidance.png'
    env = CommunicationEnv()
    critic_losses = []
    scores = []
    Q_Values = []

    n_steps = 0
    # N = 20
    learn_iters = 0
    avg_score = 0.
    best_score = 0.

    for e in range(agent.load_episode, 10001):

        # s = env.reset(input_ph3=False, input_phj=False)
        s = env.reset()
        score = 0.
        step = 0
        done = False

        np.savetxt("./Total_reward.txt",scores, delimiter=",")
        # np.savetxt("./critic_loss.txt",critic_losses, delimiter=",")
        # np.savetxt("./Q_value.txt",Q_Values, delimiter=",")


        # if e % 10 == 0 :
        #     torch.save(agent.eval_net.state_dict(), agent.dirPath + str(e) + '.h5')

        # for t in range(6000):
        while not done:

            a = agent.choose_action(s)

            # s_, r, done = env.step(a, input_ph3=False, input_phj=False)
            s_, r, done = env.step(a)

            agent.transition += [r, s_, done]

            agent.memory.store(*agent.transition)

            if n_steps >= agent.startTime:
                # if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            # loss = agent.C_L
            score += r
            s = s_
            n_steps += 1


        scores.append(score)
        avg_score = np.mean(scores[-100:])
        if avg_score > best_score:
            best_score = avg_score
            torch.save(agent.eval_net.state_dict(), checkpoint_file)
        print('Episode : {} Step : {} learn_iters : {} Action : {} avg_score : {} Reward : {} '.format(e, n_steps ,learn_iters ,a, avg_score, r))

            # Q_Values.append(agent.Q_V)
            # critic_losses.append(agent.C_L)

            # avg_score = np.mean(scores[-100:])
            # print('Episode : {} Step : {} learn_iters : {} Action : {} avg_score : {} Reward : {} Loss : {}'.format(e, n_steps ,learn_iters ,a, avg_score, r, loss))


            # if done:
            #     if avg_score > best_score:
            #         best_score = avg_score
            #         torch.save(agent.eval_net.state_dict(), checkpoint_file)

            #     scores.append(score)
            #     Q_Values.append(agent.Q_V)
            #     critic_losses.append(agent.C_L)
            #     avg_score = np.mean(scores[-100:])
            #     print('Episode : {} Step : {} learn_iters : {} Action : {} avg_score : {} Reward : {} Loss : {}'.format(e, n_steps ,learn_iters ,a, avg_score, r, loss))

            #     break



        # print('|============================================================================================|')
        # print('|=========================================  Result  =========================================|')
        # print('|                                     Total_Step : {}  '.format(n_steps))
        # print('|                      Episode : {} Total_Reward : {} '.format(e, score))
        # print('|============================================================================================|')

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(x, scores, figure_file)
    print('Finish.')