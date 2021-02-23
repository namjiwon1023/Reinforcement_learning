from gym_torcs import TorcsEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

from DDPG import Agent
from OU import OU

EPISODE_COUNT = 50000
MAX_STEPS = 100000
EXPLORE = 5000000.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Setting Neural Network Parameters
    params = {
                'memory_size' : 100000,
                'batch_size' : 64,
                'state_size' : 3,
                'action_size' : 3,
                'gamma' : 0.99,
                'tau' : 1e-3,
                'vision' : True,
                'actor_lr' : 0.000025,
                'critic_lr' : 0.00025,
                'epsilon' : 1,
                'load_model' : False,
                'load_episode' : 0,
                'train' : True,
        }

    agent = Agent(**params)

    # Ornstein-Uhlenbeck Process
    OU = OU()

    # Environment Setting
    env = TorcsEnv(vision = agent.vision, throttle = True, gear_change = False)
    param_dictionary = dict()

    actor_losses = []
    critic_losses = []
    scores = []

    n_step = 0
    N = 20
    learn_iters = 0

    # Train Process
    for e in range(agent.load_episode, EPISODE_COUNT):

        if e % 3 == 0 :
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        # s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        s = ob.img

        score = 0.
        np.savetxt("./Total_scores.txt",scores, delimiter=",")
        np.savetxt("./actor_losses.txt",actor_losses, delimiter=",")
        np.savetxt("./critic_losses.txt",critic_losses, delimiter=",")

        # Store Neural Network Parameters
        if e % 100 == 0:
            torch.save(agent.actor_eval.state_dict(), agent.Actor_dirPath + str(e) + '.h5')
            torch.save(agent.critic_eval.state_dict(), agent.Critic_dirPath + str(e) + '.h5')
            with open(agent.dirPath + str(e) + '.pkl' , 'wb') as outfile:
                if agent.epsilon is not None:
                    pickle.dump(param_dictionary, outfile)
                else:
                    pass

        for i in range(MAX_STEPS):

            loss = 0
            agent.epsilon -= 1.0 / EXPLORE
            noise = np.zeros([1, agent.action_size])
            a_n = np.zeros([1, agent.action_size])

            # Choose Action
            a = agent.actor_eval(torch.unsqueeze(torch.FloatTensor(s),0).to(device)).detach().cpu().numpy()

            # Setting Noise Functions
            if agent.train is True:
                noise[0][0] = max(agent.epsilon, 0) * OU.function(a[0][0], 0.0, 0.60, 0.30)
                noise[0][1] = max(agent.epsilon, 0) * OU.function(a[0][1], 0.5, 1.00, 0.10)
                noise[0][2] = max(agent.epsilon, 0) * OU.function(a[0][2], -0.1, 1.00, 0.05)

                # if random.random() <= 0.1:
                #     print("apply the brake")
                #     noise[0][2] = max(agent.epsilon, 0) * OU.function(a[0][2], 0.2, 1.00, 0.10)
            else:
                pass

            a_n[0][0] = a[0][0] + noise[0][0]
            a_n[0][1] = a[0][1] + noise[0][1]
            a_n[0][2] = a[0][2] + noise[0][2]

            ob, r, done, _ = env.step(a_n[0])

            s_ = ob.img

            agent.transition = [s, a_n[0], r, s_, done]
            agent.memory.store(*agent.transition)
            if n_step % N == 0:
                agent.learn()
                learn_iters += 1

            loss = agent.critic_L
            score += r
            s = s_

            print('Episode : {} Step : {} learn_iters : {} Action : {} Reward : {} Loss : {}'.format(e, n_step , learn_iters, a_n, r, loss))
            n_step += 1

            if done :

                scores.append(score)
                actor_losses.append(agent.actor_L)
                critic_losses.append(agent.critic_L)

                param_keys = ['epsilon']
                param_value = [agent.epsilon]
                param_dictionary = dict(zip(param_keys,param_value))

                break

        print('|============================================================================================|')
        print('|=========================================  Result  =========================================|')
        print('|                                     Total_Step : {}  '.format(n_step))
        print('|                      Episode : {} Total_Reward : {} '.format(e, score))
        print('|============================================================================================|')

    env.end()
    print('Finish.')
