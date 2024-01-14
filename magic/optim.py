from magic import Tensor
from magic import Module

class Adam:
    """ Adam optimizer """

    def __init__(self, model: Module, lr: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        self.dists = self.model.guide()
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {n: d.params for n, d in self.dists.items()}
        self.v = {n: d.params for n, d in self.dists.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for name, params in self.dists.items():
            for i, param in enumerate(params):
                self.m[name][i] = self.beta_1 * self.m[name][i] + (1 - self.beta_1) * param.grad
                self.v[name][i] = self.beta_2 * self.v[name][i] + (1 - self.beta_2) * (param.grad ** 2)
                m_hat = self.m[name][i] / (1 - self.beta_1 ** self.t)
                v_hat = self.v[name][i] / (1 - self.beta_2 ** self.t)
                param = param - self.lr * m_hat / (v_hat ** 0.5 + self.epsilon)

    def zero_grad(self):
        for name, params in self.dists.items():
            for param in params:
                param.grad = 0
