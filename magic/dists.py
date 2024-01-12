from magic.tensor import Tensor

import jax.numpy as np

class Normal:
    """ normal distribution """

    def __init__(self, mean: int = 0, std: int = 1) -> None:
        self.mean = mean
        self.std = std

    def log_density(self, x: Tensor) -> Tensor:
        const = Tensor(np.log(np.sqrt(2 * np.pi)))
        sigma = self.std ** 2
        return -((x - self.mean) ** 2) / (2 * sigma) - Tensor(np.log(self.std.data)) - const

    def sample(self, shape: tuple = (1,)):
        return self.mean + self.std * Tensor(np.random.normal(0, 1, size = shape))
