import os
import torch
import random
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from segment_tree import MinSegmentTree, SumSegmentTree

class ReplayBuffer:
    def __init__(self, obs_dim, max_size, batch_size=32):
        self.obs_buf = np.zeros([max_size, obs_dim],dtype=np.float32)
        self.next_obs_buf = np.zeros([max_size, obs_dim],dtype=np.float32)
        self.acts_buf = np.zeros([max_size],dtype=np.float32)
        self.rews_buf = np.zeros([max_size],dtype=np.float32)
        self.done_buf = np.zeros([max_size],dtype=np.float32)
        self.max_size, self.batch_size = max_size, batch_size
        self.ptr, self.size, = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        index = np.random.choice(self.size, self.batch_size, replace = False)
        return dict(obs = self.obs_buf[index],
                    next_obs = self.next_obs_buf[index],
                    acts = self.acts_buf[index],
                    rews = self.rews_buf[index],
                    done = self.done_buf[index])

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, max_size, batch_size=32, alpha=0.6):    # alpha: 控制采样在均匀还是贪婪的偏好 0： 均匀采样 1：贪婪采样 不改变优先级的单调性， 只是适当的调高TD-ERROR（低）的优先级
        assert alpha >= 0
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, max_size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0   # max_priority (float): max priority 最高优先级 | tree_ptr (int): next index of tree 树的下一个索引
        self.alpha = alpha
        # capacity must be positive and a power of 2.  容量必须为正且为2的幂。
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2        # [1, 2, 4, 8, 16, ... , 2**n]
        self.sum_tree = SumSegmentTree(tree_capacity)   # 根据树的容量， 生成sum_tree
        self.min_tree = MinSegmentTree(tree_capacity)   # 根据树的容量， 生成min_tree

    def store(self, obs, act, rew, next_obs, done):
        """ Store experience and priority.  储存 经验 和 优先级 """
        super().store(obs, act, rew, next_obs, done)    # 调用 ReplayBuffer 中的 store 类
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size   # 与 ReplayBuffer 中的 ptr 是一样的， 只不过 这里是在tree中

    def sample_batch(self, beta=0.4):      # beta 用来决定你有多大的程度想抵消 Prioritized Experience Replay 对收敛结果的影响。 如果 beta = 1 则代表完全抵消掉了影响， 这种情况相当于普通的ReplayBuffer
        """ Sample a batch of experiences. 抽样一批经验 """
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(obs=obs,
                    next_obs=next_obs,
                    acts=acts,
                    rews=rews,
                    done=done,
                    weights=weights,
                    indices=indices,
                    )

    def update_priorities(self, indices, priorities):             # 在训练的时候 首先 先采样， 第二进行NN的参数更新 ， 第三更新优先级P（priorities）
        """ Update priorities of sampled transitions. 更新采样过渡的优先级。 """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        """ Sample indices based on proportions. 基于比例的样本指数。"""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1) # 计算顶点的数值
        segment = p_total / self.batch_size   # 所有的P值 按照 batch_size 进行分类

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx, beta):
        """ Calculate the weight of the experience at idx. 计算idx上体验的权重。"""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

# PER parameters
# alpha: float = 0.2,
# beta: float = 0.6,
# prior_eps: float = 1e-6,

# self.memory = PrioritizedReplayBuffer(
#             obs_dim, memory_size, batch_size, alpha
#         )

# PER needs beta to calculate weights
# samples = self.memory.sample_batch(self.beta)
# weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
# indices = samples["indices"]
# NN update
# loss = torch.mean(elementwise_loss * weights)
# self.optimizer.zero_grad()
# # loss.backward()
# # self.optimizer.step()

# PER: update priorities
# loss_for_prior = elementwise_loss.detach().cpu().numpy()
# new_priorities = loss_for_prior + self.prior_eps
# self.memory.update_priorities(indices, new_priorities)
