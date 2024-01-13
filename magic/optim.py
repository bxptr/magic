from magic import Tensor

class Adam:
    def __init__(self, params, lr: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.t = 0
        self.m = {param: 0 for param in self.params}
        self.v = {param: 0 for param in self.params}

    def step(self) -> None:
        self.t += 1
        for param in self.params:
            if self.params[param].requires_grad:
                self.m[param] = self.beta_1 * self.m[param] + (1 - self.beta_1) * self.params[param].grad
                self.v[param] = self.beta_2 * self.v[param] + (1 - self.beta_2) * (self.params[param].grad ** 2)
                m_hat = self.m[param] / (1 - self.beta_1 ** self.t)
                v_hat = self.v[param] / (1 - self.beta_2 ** self.t)
                self.params[param].data -= self.lr * m_hat / (v_hat ** 0.5 + self.epsilon)

    def zero_grad(self) -> None:
        for param in self.params:
            if self.params[param].requires_grad:
                self.params[param].grad = 0

