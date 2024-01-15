import jax.numpy as np

from typing import Union

class Tensor:
    """ basic tensor """

    def __init__(self, data: object) -> "Tensor":
        if isinstance(data, Tensor):
            self.data = np.array(data.data)
            self.grad = data.grad
            self.requires_grad = data.requires_grad
        else:
            self.data = np.array(data)
            self.grad = 0
            self.requires_grad = False

        self.shape = self.data.shape

    def __radd__(self, x: Union["Tensor", float]) -> "Tensor":
        return self.__add__(x)

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

    def __pow__(self, x: Union["Tensor", float]) -> "Tensor":
        if isinstance(x, Tensor):
            return Tensor(self.data ** x.data)
        else:
            return Tensor(self.data ** x)

    def __getitem__(self, idx: int) -> "Tensor":
        return Tensor(self.data[idx])

    def __neg__(self) -> "Tensor":
        return Tensor(-self.data)

    def __abs__(self) -> "Tensor":
        return Tensor(np.abs(self.data))

    def __repr__(self):
        return f"Tensor({self.data})"
