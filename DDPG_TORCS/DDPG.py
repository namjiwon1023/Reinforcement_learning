import copy
import random
from gym_torcs import TorcsEnv
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from Actor import Actor
from Critic import Critic
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer

EPISODE_COUNT = int(2e3)
MAX_STEPS = int(1e5)
EXPLORE = 100000.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.memory = ReplayBuffer(self.state_size, self.memory_size, self.batch_size)
        self.transition = list()

        self.actor_eval = Actor(self.state_size).to(device)
        self.actor_eval.apply(init_weights)
        self.actor_target = Actor(self.state_size).to(device)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.actor_target.eval()

        self.critic_eval = Critic(self.state_size, self.action_size).to(device)
        self.critic_target = Critic(self.state_size, self.action_size).to(device)
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.critic_target.eval()

        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.critic_lr)

        self.critic_loss_func = nn.MSELoss()
        self.critic_L = 0.

        self.noise_steering = OUNoise(size = 1, mu = 0.0, theta = 0.6, sigma = 0.30)
        self.noise_acceleration = OUNoise(size = 1, mu = 0.5, theta = 1.0, sigma = 0.10)
        self.noise_brake = OUNoise(size = 1, mu = -0.1, theta = 1.0, sigma = 0.05)

        self.dirPath = './load_state/DDPG_epsilon_'
        self.Actor_dirPath = './load_state/DDPG_actor_'
        self.Critic_dirPath = './load_state/DDPG_critic_'

        if self.load_model :
            self.actor_eval.load_state_dict(torch.load(self.Actor_dirPath + str(self.load_episode) + '.h5'))
            self.critic_eval.load_state_dict(torch.load(self.Critic_dirPath + str(self.load_episode) + '.h5'))
            with open(self.dirPath + str(self.load_episode) + '.pkl') as outfile:
                param = pickle.load(outfile)
                self.epsilon = param.get('epsilon')

    def learn(self):
        samples = self.memory.sample_batch()

        state = torch.FloatTensor(samples['obs']).to(device)
        next_state = torch.FloatTensor(samples['next_obs']).to(device)
        action = torch.FloatTensor(samples['act']).reshape(-1,3).to(device)
        reward = torch.FloatTensor(samples['rew']).reshape(-1,1).to(device)
        done = torch.FloatTensor(samples['done']).reshape(-1,1).to(device)

        # Critic Network Update
        mask = 1 - done
        next_action = self.actor_target(next_state).to(device)
        next_value = self.critic_target(next_state, next_action).to(device)
        target_values = (reward + self.gamma * next_value * mask).to(device)

        eval_values = self.critic_eval(state, action).to(device)
        critic_loss = self.critic_loss_func(eval_values, target_values).to(device)
        self.critic_L = critic_loss.detach().cpu().numpy()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Network Update
        actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean().to(device)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._target_soft_update()

    def _target_soft_update(self):
        tau = self.tau

        for target_params , eval_params in zip(
            self.actor_target.parameters(), self.actor_eval.parameters()
            ):
            target_params.data.copy_(tau * eval_params.data + (1.0 - tau) * target_params.data)

        for target_params , eval_params in zip(
            self.critic_target.parameters(), self.critic_eval.parameters()
            ):
            target_params.data.copy_(tau * eval_params.data + (1.0 - tau) * target_params.data)

if __name__ == "__main__":
    params = {
                'memory_size' : int(1e5),
                'batch_size' : 32,
                'state_size' : 29,
                'action_size' : 3,
                'gamma' : 0.95,
                'tau' : 1e-3,
                'vision' : False,
                'actor_lr' : 1e-4,
                'critic_lr' : 1e-3,
                'epsilon' : 1,
                'load_model' : False,
                'load_episode' : 0,
                'step' : 0,
                'train_start' : 32,
        }
    agent = Agent(**params)
    env = TorcsEnv(vision = agent.vision, throttle = True, gear_change = False)
    param_dictionary = dict()

    for e in range(agent.load_episode, EPISODE_COUNT):
        if np.mod(e, 3) == 0 :
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()
        print("ob:",ob)
        '''ob: Observaion(focus=array([0.17946899, 0.17301649, 0.16678551, 0.160778  , 0.1549965 ],dtype=float32),
        speedX=-0.00010545899470647176,
        speedY=0.00011448666453361511,
        speedZ=-0.0002021526669462522,
        angle=-0.00027785650014532017,
        damage=array(0., dtype=float32),
        opponents=array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1.], dtype=float32),
        rpm=0.094247802734375,
            track=array([0.0564915 , 0.316134  , 0.254871  , 0.21488701, 0.193023  ,
                                    0.1827775 , 0.17751051, 0.17301649, 0.16987301, 0.16678551,
                                    0.16375351, 0.160778  , 0.156707  , 0.15218951, 0.1441075 ,
                                    0.1294325 , 0.1091015 , 0.08792149, 0.0508225 ], dtype=float32),
        trackPos=0.0020843499805778265,
        wheelSpinVel=array([0.      , 0.      , 0.      , 0.303038], dtype=float32))
'''


        s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        print("s.shape:",s.shape) # s.shape: (29,)
        print("s.shape[0]",s.shape[0]) # s.shape[0]: 29
        print("s:",s)
        '''s: [-2.77856500e-04  5.64914979e-02  3.16134006e-01  2.54871011e-01
            2.14887008e-01  1.93022996e-01  1.82777494e-01  1.77510515e-01
            1.73016489e-01  1.69873014e-01  1.66785508e-01  1.63753510e-01
            1.60778001e-01  1.56707004e-01  1.52189508e-01  1.44107506e-01
            1.29432499e-01  1.09101497e-01  8.79214928e-02  5.08225001e-02
            2.08434998e-03 -1.05458995e-04  1.14486665e-04 -2.02152667e-04
            0.00000000e+00  0.00000000e+00  0.00000000e+00  3.03038000e-03
            9.42478027e-02]
'''



        total_reward = 0.
        done = False

        if e % 10 == 0:
            torch.save(agent.actor_eval.state_dict(), agent.Actor_dirPath + str(e) + '.h5')
            torch.save(agent.critic_eval.state_dict(), agent.Critic_dirPath + str(e) + '.h5')
            with open(agent.dirPath + str(e) + '.pkl' , 'wb') as outfile:
                pickle.dump(param_dictionary, outfile)

        for i in range(MAX_STEPS):
            loss = 0
            agent.epsilon -= 1.0 / EXPLORE
            noise = np.zeros([1, agent.action_size])
            a_n = np.zeros([1, agent.action_size])

            # a = agent.actor_eval(torch.tensor(s.reshape(1, s.shape[0]), device=device).float()).detach().cpu().numpy()
            a = agent.actor_eval(torch.unsqueeze(torch.FloatTensor(s),0), device=device).detach().cpu().numpy()
            print("a.shape:",a.shape)   # a.shape: (1, 3)
            print("a",a) # a [[2.6901831e-12 5.0000000e-01 5.0000000e-01]]
            noise[0][0] = agent.noise_steering.sample() * max(agent.epsilon, 0)
            noise[0][1] = agent.noise_acceleration.sample() * max(agent.epsilon, 0)
            noise[0][2] = agent.noise_brake.sample() * max(agent.epsilon, 0)
            print(noise.shape)  # (1, 3)
            print(noise)  # [[ 0.01292339  0.59637893 -0.08812192]]

            a_n[0][0] = a[0][0] + noise[0][0]
            a_n[0][1] = a[0][1] + noise[0][1]
            a_n[0][2] = a[0][2] + noise[0][2]
            print("a_n.shape:",a_n.shape)  # a_n.shape: (1, 3)
            print("a_n:",a_n)  # a_n: [[0.01292339 1.09637893 0.41187808]]

            ob, r, done, info = env.step(a_n[0])

            s_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            print("s_.shape:",s_.shape)   # s_.shape: (29,)
            print("s_:",s_)
            '''s_: [-2.24125915e-04  5.64539991e-02  3.16083997e-01  2.54840493e-01
                2.14871496e-01  1.93016484e-01  1.82775989e-01  1.77511007e-01
                1.73019007e-01  1.69876993e-01  1.66791007e-01  1.63760513e-01
                1.60786003e-01  1.56716496e-01  1.52201504e-01  1.44122496e-01
                1.29454002e-01  1.09129496e-01  8.79549980e-02  5.08549958e-02
                2.90324003e-03 -4.90586658e-04  4.52576677e-04 -1.41160004e-04
                2.80751996e-02  6.15701033e-03  2.00121999e-02  2.00449992e-02
                9.42478027e-02]
                '''
            agent.transition = [s, a_n[0], r, s_, done]
            agent.memory.store(*agent.transition)
            if agent.memory.__len__() > agent.train_start:
                agent.learn()

            agent.step += 1
            loss += agent.critic_L
            total_reward += r
            s = s_

            if done :
                print('Episode : {} Step : {}  Action : {} Reward : {} Loss : {}'.format(e, agent.step , a_n, r, loss))
                param_keys = ['epsilon']
                param_value = [agent.epsilon]
                param_dictionary = dict(zip(param_keys,param_value))
                break
        print('-------------------------------------------------------------------------------------------')
        print('Episode : {} Total_Reward : {} '.format(e, total_reward))
        print('Total_Step : {}  '.format(agent.step))
        print('-------------------------------------------------------------------------------------------')

    env.end()
    print('Finish.')