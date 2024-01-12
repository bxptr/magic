import jax.numpy as np

from typing import Union

class Tensor:
    """ basic tensor """

    def __init__(self, data: float) -> "Tensor":
        self.data = jax.device_put(data)

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
