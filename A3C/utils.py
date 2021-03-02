#!usr/bin/python
#-*- coding:utf-8 -*-

import torch as T
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp

def v_wrap(np_array, dtype=np.float32):            # 判定输入的数据类型， 如果不是设定的数据类型， 进行数据类型强制转化。
    if np_array.dtype != dtype:                    # 输入的类型如果不是np.float32的情况下， 强制转换为np.float32
        np_array = np_array.astype(dtype)
    return T.from_numpy(np_array)                  # 转换成torch.tensor

def set_init(layers):                               # 基本的神经网络初始方法
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)    # 设置神经网络层的权重为以均值为0，方差为0.1的，正态分布取值
        nn.init.constant_(layer.bias, 0.)          # 填充神经网络层的bias为0.

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):             # 需要输入的数据： 优化器， 下一个状态， 本地和全局的网络， 终止， batchsize的状态、动作以及奖励值， 衰减率
    if done:
        v_s_ = 0.               # 终止时 V(s') 为0
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]      # 将 下一个状态的值进行数据转化 输入本地的神经网络中

    buffer_v_target = []                   # critic更新时的 target_v的值， 计算公式： target_V = r + gamma*target(V(s'))

    for r in br[::-1]:             # 进行反向排序
        v_s_ = r + gamma*v_s_      # 这个地方 不应该是 V(s) 吗？？
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    # 进行本地网络的loss计算, 并计算本地的梯度                                                       # np.hstack(tup) ： 沿着水平方向将数组堆叠起来。
    loss = lnet.loss_func(v_wrap(np.vstack(bs)),                                # np.vstack(tup) : 沿着竖直方向将矩阵堆叠起来， 除开第一维外，被堆叠的矩阵各维度要一致。
                        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
                        v_wrap(np.array(buffer_v_target)[:, None]))

    # 计算局部梯度并将局部参数推入全局
    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):        # 讲本地的网络参数 赋值给 全局网络
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    # 获取全局参数
    lnet.load_state_dict(gnet.state_dict())

def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():                                                    # 多线程共享数据， 默认共享对象内部的关联锁是递归锁（默认），使用 with counter.get_lock():
        global_ep.value += 1                                                      #                                                           counter.value += 1
    with global_ep_r.get_lock():                                                  # 的方式进行运算
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value*0.99 + ep_r*0.01
    res_queue.put(global_ep_r.value)                                         # 将 全局的episode的reward进行储存

    print(name,
        "Ep : ", global_ep.value,
        "| Ep_r : %.0f" % global_ep_r.value,)