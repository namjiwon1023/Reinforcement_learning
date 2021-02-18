import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from CriticNet import CriticNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDQN(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.eval_net = Network(self.state_size, self.hidden_size, self.action_size).to(device)
        self.target_net = Network(self.state_size, self.hidden_size, self.action_size).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.update_time = update_time
        self.update_counter = 0

        self.memory = ReplayBuffer(self.memory_size, self.batch_size)
        self.transition = list()

        self.optimizer = optim.SGD(self.eval_net.parameters(),lr=self.lr)
        self.loss = nn.MSELoss()

        self.C_L = 0.
        self.Q_V = 0.

        self.dirPath = './load_state/DDQN_critic_'

        if self.load_model :
            self.eval_net.load_state_dict(torch.load(self.dirPath + str(self.load_episode) + '.h5'))

    def choose_action(self, state):

        if self.epsilon > np.random.random():
            choose_action = np.random.randint(0,self.action_size)
        else :
            choose_action = self.eval_net(torch.unsqueeze(torch.FloatTensor(state),0).to(device)).argmax()
            choose_action = choose_action.detach().cpu().numpy()
        self.transition = [state, choose_action]
        return choose_action

    def target_net_update(self):

        if self.update_counter % self.update_time == 0 :
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.update_counter += 1


    def learn(self):

        self.target_net_update()
        samples = self.memory.sample_batch()

        state = torch.FloatTensor(samples['obs']).to(device)
        next_state = torch.FloatTensor(samples['next_obs']).to(device)
        action = torch.LongTensor(samples['act']).reshape(-1,6).to(device)
        reward = torch.FloatTensor(samples['rew']).reshape(-1,1).to(device)
        done = torch.FloatTensor(samples['done']).reshape(-1,1).to(device)

        curr_q = self.eval_net(state).gather(1, action)
        self.Q_V = curr_q.detach().cpu().numpy()
        next_q = self.target_net(next_state).gather(1, self.eval_net(next_state).argmax(dim = 1, keepdim = True)).detach()

        mask = 1 - done

        target_q = (reward + self.gamma * next_q * mask).to(device)

        loss = self.loss(curr_q, target_q)
        self.C_L = loss.detach().cpu().numpy()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":

    params = {
                'memory_size' : 2000,
                'batch_size' : 5,
                'state_size' : 3,
                'action_size' : 6,
                'epsilon' : 0.1,
                'update_time' : 10,
                'gamma' : 0.4,
                'lr' : 0.1,
                'load_model' : False,
                'load_episode' : 0,
        }

    agent = DDQN(**params)

    env = Env()
    critic_losses = []
    scores = []
    Q_Values = []

    for e in range(agent.load_episode, 10000):

        s = env.reset()
        score = 0.
        step = 0

        np.savetxt("./Total_reward.txt",scores, delimiter=",")
        np.savetxt("./critic_loss.txt",critic_losses, delimiter=",")
        np.savetxt("./Q_value.txt",Q_Value, delimiter=",")

        if e % 10 == 0 :
            torch.save(agent.eval_net.state_dict(), agent.dirPath + str(e) + '.h5')

        for t in range(6000):
        # while not done:

            a = agent.choose_action(s)

            s_, r, done = env.step(a)

            agent.transition += [r, s_, done]

            agent.memory.store(*agent.transition)

            agent.learn()

            loss = agent.C_L
            score += r
            s = s_

            print('Episode : {} Step : {}  Action : {} Reward : {} Loss : {}'.format(e, step , a, r, loss))
            agent.step += 1

            if done:

                scores.append(score)
                Q_Values.append(agent.Q_V)
                critic_losses.append(agent.C_L)
                break

        print('|============================================================================================|')
        print('|=========================================  Result  =========================================|')
        print('|                                     Total_Step : {}  '.format(step))
        print('|                      Episode : {} Total_Reward : {} '.format(e, score))
        print('|============================================================================================|')

    print('Finish.')