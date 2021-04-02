import copy
import random
import os
from gym_torcs import TorcsEnv
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Actor import ActorNetwork
from Critic import CriticNetwork
from ReplayBuffer import ReplayBuffer

if T.backends.cudnn.enabled:
    T.backends.cudnn.benchmark = False
    T.backends.cudnn.deterministic = True
seed = 123
T.cuda.manual_seed(seed)
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
print('Using GPU : ', T.cuda.is_available() , ' |  Seed : ', seed)


def _layer_norm(m, std=1.0, bias_const=1e-6):   # bias_const = 1e-6
    if isinstance(m, nn.Linear):
        T.nn.init.orthogonal_(m.weight, std)
        T.nn.init.constant_(m.bias, bias_const)

class SACAgent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        dirPath='/home/nam/Reinforcement_learning/SAC_TORCS_cnn_add_sensors'
        self.checkpoint = os.path.join(dirPath,'alpha_optimizer')

        self.in_dims = 3
        self.n_sensors = 29
        self.n_actions = 3

        self.memory = ReplayBuffer(self.memory_size, self.in_dims, self.n_actions, self.n_sensors, self.batch_size)

        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.actor = ActorNetwork(self.in_dims, self.n_actions, self.learning_rate, self.n_sensors)
        self.actor.apply(_layer_norm)

        self.critic_eval = CriticNetwork(self.in_dims, self.n_actions, self.learning_rate, self.n_sensors)
        self.critic_eval.apply(_layer_norm)
        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.transition = list()

    def choose_action(self, state, sensor):

        action, _ = self.actor(T.unsqueeze(T.FloatTensor(state),0).to(self.actor.device), T.unsqueeze(T.FloatTensor(sensor),0).to(self.actor.device))

        action = action.cpu().detach().numpy()
        self.transition = [state, sensor, action[0]]
        return action

    def target_soft_update(self):
        tau = self.tau
        with T.no_grad():
            for t_p, l_p in zip(self.critic_target.parameters(), self.critic_eval.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def learn(self):
        samples = self.memory.sample_batch()
        state = T.FloatTensor(samples["state"]).to(self.actor.device)
        next_state = T.FloatTensor(samples["next_state"]).to(self.actor.device)
        sensor_state = T.FloatTensor(samples["sensor_state"]).to(self.actor.device)
        next_sensor_state = T.FloatTensor(samples["next_sensor_state"]).to(self.actor.device)
        action = T.FloatTensor(samples["action"]).reshape(-1, self.n_actions).to(self.actor.device)
        reward = T.FloatTensor(samples["reward"]).reshape(-1, 1).to(self.actor.device)
        done = T.FloatTensor(samples["done"]).reshape(-1, 1).to(self.actor.device)
        mask = (1 - done).to(self.actor.device)

        # critic update
        with T.no_grad():
            next_action, next_log_prob = self.actor(next_state, next_sensor_state)
            q1_target, q2_target = self.critic_target(next_state, next_action, next_sensor_state)
            q_target = T.min(q1_target, q2_target)
            value_target = reward + self.GAMMA * (q_target - self.alpha * next_log_prob) * mask
        q1_eval, q2_eval = self.critic_eval(state, action, sensor_state)
        critic_loss = F.mse_loss(q1_eval, value_target) + F.mse_loss(q2_eval, value_target)

        self.critic_eval.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval.optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        new_action, new_log_prob = self.actor(state, sensor_state)
        q_1, q_2 = self.critic_eval(state, new_action, sensor_state)
        q = T.min(q_1, q_2)
        actor_loss = (self.alpha * new_log_prob - q).mean()
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = True

        self.alpha = self.log_alpha.exp()

        if self.total_episode % self.update_time == 0 :
            self.target_soft_update()

    def save_models(self):
        print('------ save models ------')
        self.actor.save_models()
        self.critic_eval.save_models()

        T.save(self.alpha_optimizer.state_dict(), self.checkpoint)

    def load_models(self):
        print('------ load models ------')
        self.alpha_optimizer.load_state_dict(T.load(self.checkpoint))

        self.actor.load_models()

        self.critic_eval.load_models()
        self.critic_target = copy.deepcopy(self.critic_eval)

if __name__ == "__main__":

    params = {
                'GAMMA' : 0.99,
                'learning_rate' : 2e-4,
                'tau' : 0.005,
                'update_time' : 1,
                'memory_size' : int(2 ** 20),
                'batch_size' : 256,
                'total_episode' : 0,
                'vision' : True,
}
    agent = SACAgent(**params)
    sac_actor_parameter = '/home/nam/Reinforcement_learning/SAC_TORCS_cnn_add_sensors/sac_actor'
    sac_actor_optimizer_parameter = '/home/nam/Reinforcement_learning/SAC_TORCS_cnn_add_sensors/sac_actor_optimizer'
    sac_critic_parameter = '/home/nam/Reinforcement_learning/SAC_TORCS_cnn_add_sensors/sac_critic'
    sac_critic_optimizer_parameter = '/home/nam/Reinforcement_learning/SAC_TORCS_cnn_add_sensors/sac_critic_optimizer'
    alpha_optimizer_parameter = '/home/nam/Reinforcement_learning/SAC_TORCS_cnn_add_sensors/alpha_optimizer'
    reward_file = '/home/nam/Reinforcement_learning/SAC_TORCS_cnn_add_sensors/reward.txt'
    if os.path.exists(sac_actor_parameter) and os.path.exists(sac_actor_optimizer_parameter) and os.path.exists(sac_critic_parameter) and os.path.exists(sac_critic_optimizer_parameter) and os.path.exists(alpha_optimizer_parameter):
        agent.load_models()
    else:
        print('------ No parameters available! ------')


    plt.ion()
    plt.figure(figsize=(15, 5))

    EPISODE_COUNT = int(1e6)
    MAX_STEPS = 100000
    best_score = 0

    env = TorcsEnv(vision = agent.vision, throttle = True, gear_change = False)
    avg_score = 0
    scores = []

    # Train Process
    for e in range(1, EPISODE_COUNT + 1):
        agent.total_episode = e

        if e % 3 == 0 :
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        state = ob.img
        sensor_state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        step = 0
        score = 0.

        np.savetxt("./reward.txt",scores, delimiter=",")

        for i in range(MAX_STEPS):
            action = agent.choose_action(state, sensor_state)

            ob, r, done, _ = env.step(action[0])

            state_ = ob.img
            sensor_state_ = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

            agent.transition += [r, state_, sensor_state_, done]
            agent.memory.store(*agent.transition)

            agent.learn()

            step += 1
            score += r
            state = state_
            sensor_state = sensor_state_

            print('Episode : {} | Action : {} | Reward : {}'.format(e, action[0], r))
            if done :
                scores.append(score)
                print('--------------------------------------------------------------|')
                print('Episode : {} | Step : {} | Reward : {}'.format(e, step, score))
                print('--------------------------------------------------------------|')
                break

        avg_score = np.mean(scores[-10:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

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

    env.end()
    print('Finish.')
