import jax
import jax.numpy as np

from typing import Union

class Tensor:
    """ basic tensor """

    def __init__(self, data: object) -> "Tensor":
        if isinstance(data, Tensor):
            self.data = jax.device_put(data.data)
            self.grad = data.grad
            self.requires_grad = data.requires_grad
        else:
            self.data = jax.device_put(data)
            self.grad = 0
            self.requires_grad = False

    def __add__(self, x: Union["Tensor", float]) -> "Tensor":
        if isinstance(x, Tensor):
            return Tensor(self.data + x.data)
        else:
            return Tensor(self.data + x)

    def __sub__(self, x: Union["Tensor", float]) -> "Tensor":
        if isinstance(x, Tensor):
            return Tensor(self.data - x.data)
        else:
            return Tensor(self.data - x)

    def __mul__(self, x: Union["Tensor", float]) -> "Tensor":
        if isinstance(x, Tensor):
            return Tensor(self.data * x.data)
        else:
            return Tensor(self.data * x)

    def __truediv__(self, x: Union["Tensor", float]) -> "Tensor":
        if isinstance(x, Tensor):
            return Tensor(self.data / x.data)
        else:
            return Tensor(self.data / x)

    def __repr__(self):
        return f"Tensor({self.data})"
