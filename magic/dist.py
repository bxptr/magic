from magic import Tensor

import jax.numpy as np

class Distribution:
    """ base distribution class """

    def log_density(self, x: Tensor) -> Tensor:
        raise NotImplementedError("must be implemented by subclass")

    def sample(self, shape: tuple = (1,)) -> Tensor:
        raise NotImplementedError("must be implemented by subclass")

    def __call__(self, *args: object) -> Tensor:
        return self.sample(*args)

    def __repr__(self) -> str:
        attrs = [f"{key}={value}" for key, value in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(attrs)})"

class Normal(Distribution):
    """ normal distribution """

    def __init__(self, mean: int = 0, std: int = 1) -> None:
        self.mean = mean
        self.std = std

    def log_density(self, x: Tensor) -> Tensor:
        const = Tensor(np.log(np.sqrt(2 * np.pi)))
        sigma = self.std ** 2
        return -((x - self.mean) ** 2) / (2 * sigma) - Tensor(np.log(self.std.data)) - const

    def sample(self, shape: tuple = (1,)) -> Tensor:
        return self.mean + self.std * Tensor(np.random.normal(0, 1, size = shape))
