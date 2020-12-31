import copy
import random
from gym_torcs import TorcsEnv
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_size):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, 300)
        self.fc2 = nn.Linear(300, 600)

        self.steering = nn.Linear(600, 1)
        self.steering.weight.data.normal_(0,1e-4)

        self.acceleration = nn.Linear(600, 1)
        self.acceleration.weight.data.normal_(0,1e-4)

        self.brake = nn.Linear(600, 1)
        self.brake.weight.data.normal_(0,1e-4)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        steering_out = torch.tanh(self.steering(x))
        acceleration_out = torch.sigmoid(self.acceleration(x))
        brake_out = torch.sigmoid(self.brake(x))

        out = torch.cat((steering_out, acceleration_out, brake_out), 1)
        return out


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_1 = nn.Linear(state_size, 300)
        self.state_2 = nn.Linear(300, 600)

        self.action_1 = nn.Linear(action_size, 600)

        self.hidden_layer = nn.Linear(600, 600)

        self.state_value = nn.Linear(600, action_size)

    def forward(self, state, action):
        s_300 = F.relu(self.state_1(state))
        a_600 = self.action_1(action)
        s_600 = self.state_2(s_300)
        cat = s_600 + a_600
        hidden = F.relu(self.hidden_layer(cat))
        V = self.state_value(hidden)
        return V

class ReplayBuffer:

    def __init__(self, obs_dim:int, memory_size : int, batch_size : int = 32):

        self.obs_buf = np.zeros([memory_size, obs_dim], dtype = np.float32)
        self.next_obs_buf = np.zeros([memory_size, obs_dim], dtype = np.float32)
        self.acts_buf = np.zeros([memory_size], dtype = np.float32)
        self.rews_buf = np.zeros([memory_size], dtype = np.float32)
        self.done_buf = np.zeros([memory_size], dtype = np.float32)

        self.max_size, self.batch_size = memory_size, batch_size
        self.ptr, self.size, = 0,0

    def store(self,
            obs:np.ndarray,
            act : np.ndarray,
            rew:float,
            next_obs:np.ndarray,
            done:bool ):

        self.obs_buf[self.ptr]=obs
        self.next_obs_buf[self.ptr]=next_obs
        self.acts_buf[self.ptr]=act
        self.rews_buf[self.ptr]=rew
        self.done_buf[self.ptr]=done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size +1, self.max_size)

    def sample_batch(self):

        index = np.random.choice(self.size, size = self.batch_size , replace=False)

        return dict(obs = self.obs_buf[index],
                    next_obs = self.next_obs_buf[index],
                    acts = self.acts_buf[index],
                    rews = self.rews_buf[index],
                    done = self.done_buf[index])

    def __len__(self):

        return self.size

class OUNoise:

    def __init__(self,
                size : int,
                mu : float = 0.0,
                theta : float = 0.15,
                sigma : float = 0.2,
        ):
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
            )
        self.state = x + dx
        return self.state

class DDPGAgent:
	def __init__(self,memory_size : int,
                    ou_noise_theta: float,
                    ou_noise_sigma: float,
                    batch_size : int,
                    state_size : int = 29,
                    action_size : int = 3,
                    gamma : float = 0.99,
                    tau: float = 5e-3,
                    initial_random_steps: int = 1e4,
                    ):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_size = state_size
        self.action_size =  action_size

        self.batch_size = batch_size
        self.memory_size = memory_size

        self.actor_eval = Actor(self.state_size).to(self.device)
        self.actor_target = Actor(self.state_size).to(self.device)
        self.actor_target.load_state_dict(self.actor_eval.state_dict())

        self.critic_eval = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.total_step = 0

        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        self.memory = ReplayBuffer(self.state_size, self.memory_size, self.batch_size)

        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )

        self.transition = list()


	def select_action(self, state):

		if self.total_step < self.initial_random_steps:
			selected_action = np.random.randint(0,self.action_size)
		else :
			selected_action = self.actor_eval(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()

        noise = self.noise.sample()
        selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        self.transition = [state, selected_action]
		return selected_action

	def update_model(self):


        samples = self.memory.sample_batch()

        state = torch.FloatTensor(samples['obs']).to(self.device)
        next_state = torch.FloatTensor(samples['next_obs']).to(self.device)
        action = torch.LongTensor(samples['act']).reshape(-1,1).to(self.device)
        reward = torch.FloatTensor(samples['rew']).reshape(-1,1).to(self.device)
        done = torch.FloatTensor(samples['done']).reshape(-1,1).to(self.device)

		masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks

        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._target_soft_update()

    def _target_soft_update(self):
        tau = self.tau

        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor_eval.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic_eval.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def train(self, num_frames: int):

        state = self.env.reset()
        actor_losses = []
        critic_losses = []
        scores = []
        score = 0
        for self.total_step in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state = env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if (
                len(self.memory) >= self.batch_size
                and self.total_step > self.initial_random_steps
            ):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
        self.env.close()
