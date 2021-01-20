import numpy as np

# 自适应 噪声缩放
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    def reset(self):
        pass

# 高斯噪声， 前后两步都是完全独立的
class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# Ornstein-Uhlenbeck 噪声， 使用情况： 惯性系统（环境）， 保护实际机器臂/机器人 ， 自相关噪声， 后一步的噪声受到前一步的影响（且是马尔科夫的）
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# 不相关的均匀噪声，均匀（Uniform）的意思是信号的值是随机的并且服从均匀分布，也就是说在一定范围内信号出现任意值的概率 都是一样的。不相关（Uncorrelated）的意思是信号在不同时间的值是独立不相关的；也就是说， 某一时刻的信号值不能够提供其他任何时候的信号值的信息。
class UncorrelatedUniformNoise(_Noise):

    def evaluate(self, ts):
        ys = np.random.uniform(-self.amp, self.amp, len(ts))
        return ys
# signal = thinkdsp.UncorrelatedUniformNoise()
# wave = signal.make_wave(duration=0.5, framerate=11025)


# 布朗噪声，相关的，它的值是上一时刻的值加上一个随机的步长。 之所以叫布朗噪声，是由于它和布朗运动很类似。悬浮在液体上微小粒子由于受到分子的撞击产生的无规则运动，被称为 布朗运动，在数学上，可以用 随机游走（random walk） 来描述，也就是每步运动的距离服从一定的随机分布。
# 先生成一个不相关的随机步长，然后将他们累加
class BrownianNoise(_Noise):

    def evaluate(self, ts):
        dys = np.random.uniform(-1, 1, len(ts))
        ys = np.cumsum(dys)
        ys = normalize(unbias(ys), self.amp)
        return ys