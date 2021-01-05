import copy
import random
from gym_torcs import TorcsEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from Actor import Actor
from Critic import Critic
from OU import OU
from ReplayBuffer import ReplayBuffer

# Ornstein-Uhlenbeck Process
OU = OU()

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

        # memory setting
        self.memory = ReplayBuffer(self.state_size, self.memory_size, self.batch_size)
        self.transition = list()

        # Actor Network setting(eval , target)
        self.actor_eval = Actor(self.state_size).to(device)
        self.actor_eval.apply(init_weights)

        self.actor_target = Actor(self.state_size).to(device)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.actor_target.eval()

        # Critic Network setting(eval , target)
        self.critic_eval = Critic(self.state_size, self.action_size).to(device)

        self.critic_target = Critic(self.state_size, self.action_size).to(device)
        self.critic_target.load_state_dict(self.critic_eval.state_dict())
        self.critic_target.eval()

        # Optimization Setting
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.critic_lr)

        # Critic Loss Function
        self.critic_loss_func = nn.MSELoss()
        self.critic_L = 0.

        # Load Neural Network Parameters
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
        # Random Samples
        samples = self.memory.sample_batch()

        state = torch.FloatTensor(samples['obs']).to(device)
        next_state = torch.FloatTensor(samples['next_obs']).to(device)
        action = torch.FloatTensor(samples['act']).reshape(-1,3).to(device)
        reward = torch.FloatTensor(samples['rew']).reshape(-1,1).to(device)
        done = torch.FloatTensor(samples['done']).reshape(-1,1).to(device)

        # Critic Network Update
        mask = (1 - done).to(device)
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
        actor_loss = -self.critic_eval(state, self.actor_eval(state)).to(device).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update -> Target Network Parameters
        # θQ‘←τθQ+(1−τ)θQ‘
        # θμ‘←τθμ+(1−τ)θμ‘
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

    # Setting Neural Network Parameters
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
        }

    agent = Agent(**params)

    # Environment Setting
    env = TorcsEnv(vision = agent.vision, throttle = True, gear_change = False)
    param_dictionary = dict()

    # Train Process
    for e in range(agent.load_episode, EPISODE_COUNT):

        if e % 3 == 0 :
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        s = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward = 0.
        done = False

        # Store Neural Network Parameters
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

            # Choose Action
            a = agent.actor_eval(torch.unsqueeze(torch.FloatTensor(s),0).to(device)).detach().cpu().numpy()

            # Setting Noise Functions
            noise[0][0] = max(agent.epsilon, 0) * OU.function(a[0][0], 0.0, 0.60, 0.30)
            noise[0][1] = max(agent.epsilon, 0) * OU.function(a[0][1], 0.5, 1.00, 0.10)
            noise[0][2] = max(agent.epsilon, 0) * OU.function(a[0][2], -0.1, 1.00, 0.05)

            if random.random() <= 0.1:
                print("apply the brake")
                noise[0][2] = max(agent.epsilon, 0) * OU.function(a[0][2], 0.2, 1.00, 0.10)

            a_n[0][0] = a[0][0] + noise[0][0]
            a_n[0][1] = a[0][1] + noise[0][1]
            a_n[0][2] = a[0][2] + noise[0][2]

            ob, r, done, _ = env.step(a_n[0])

            s_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

            agent.transition = [s, a_n[0], r, s_, done]
            agent.memory.store(*agent.transition)

            agent.learn()

            loss += agent.critic_L
            total_reward += r
            s = s_

            print('Episode : {} Step : {}  Action : {} Reward : {} Loss : {}'.format(e, agent.step , a_n, r, loss))
            agent.step += 1

            if done :
                param_keys = ['epsilon']
                param_value = [agent.epsilon]
                param_dictionary = dict(zip(param_keys,param_value))
                break
        print('-------------------------------------Result------------------------------------------------')
        print('---------------------------Episode : {} Total_Reward : {} ---------------------------------'.format(e, total_reward))
        print('---------------------------Total_Step : {}  -----------------------------------------------'.format(agent.step))
        print('-------------------------------------------------------------------------------------------')

    env.end()
    print('Finish.')