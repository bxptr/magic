from magic import Tensor

import jax
import jax.numpy as np
import numpy

from typing import Union

class Distribution:
    """ base distribution class """

    def _check_inputs(self, *args: object) -> None:
        checked_args = []
        for arg in args:
            if isinstance(arg, Distribution):
                positive = False
                while not positive:
                    sample = arg.sample()
                    if sample.data > 0:
                        checked_args.append(sample)
                        positive = True
            else:
                checked_args.append(arg)
        return checked_args

    def log_density(self, x: Tensor) -> Tensor:
        raise NotImplementedError("must be implemented by subclass")

    def sample(self, shape: tuple = (1,)) -> Tensor:
        raise NotImplementedError("must be implemented by subclass")

    def __call__(self, *args: object) -> Tensor:
        return self.sample(*args)

    def __add__(self, x: Union[Tensor, float]) -> Tensor:
        if isinstance(x, Tensor):
            return Tensor(self.sample(x.shape) + x.data)
        else:
            return Tensor(self.sample() + x)

    def __sub__(self, x: Union[Tensor, float]) -> Tensor:
        if isinstance(x, Tensor):
            return Tensor(self.sample(x.shape) - x.data)
        else:
            return Tensor(self.sample() - x)

    def __mul__(self, x: Union[Tensor, float]) -> Tensor:
        if isinstance(x, Tensor):
            return Tensor(self.sample(x.shape) * x.data)
        else:
            return Tensor(self.sample() * x)

    def __truediv__(self, x: Union[Tensor, float]) -> Tensor:
        if isinstance(x, Tensor):
            return Tensor(self.sample(x.shape)  / x.data)
        else:
            return Tensor(self.sample() / x)

    def __pow__(self, x: Union[Tensor, float]) -> Tensor:
        if isinstance(x, Tensor):
            return Tensor(self.sample(x.shape) ** x.data)
        else:
            return Tensor(self.sample() ** x)

    def __neg__(self) -> Tensor:
        return Tensor(-self.sample())

    def __abs__(self) -> Tensor:
        return Tensor(np.abs(self.sample()))

    def __repr__(self) -> str:
        attrs = [f"{key}={value}" for key, value in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(attrs)})"

class Normal(Distribution):
    """ normal distribution """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        mean, std = self._check_inputs(mean, std)
        self.mean = Tensor(mean)
        self.std = Tensor(std)
        self.params = [self.mean, self.std]
        self.key = jax.random.PRNGKey(0)

    def log_density(self, x: Tensor) -> Tensor:
        const = Tensor(np.log(np.sqrt(2 * np.pi)))
        sigma = self.std ** 2
        return (-((x - self.mean) ** 2) / (sigma * 2) - Tensor(np.log1p(self.std.data)) - const)[0]

    def sample(self, shape: tuple = (1,)) -> Tensor:
        self.key, subkey = jax.random.split(self.key)
        return self.mean + self.std * Tensor(jax.random.normal(subkey, shape))
