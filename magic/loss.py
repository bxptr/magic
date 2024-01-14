from magic import Tensor
from magic import Module

import jax
import jax.numpy as np

class LossTensor(Tensor):
    def __init__(self, data: object, loss: "Loss") -> None:
        super().__init__(data)
        self._loss = loss

    def backward(self) -> None:
        return jax.grad(self._loss._jax, argnums = 0, allow_int = True)(*self._loss.last.values())

class Loss:
    """ base loss class """

    def __init__(self) -> None:
        self.last = {}

    def _set_last(self, **data: object) -> Tensor:
        self.last = data

    def _jax(self) -> np.array:
        return self().data

    def __call__(self, *args: object) -> Tensor:
        raise NotImplementedError("must be implemented by subclass")

class ELBO(Loss):
    """ ELBO loss """

    def __call__(self, model: Module, y: Tensor, *x: Tensor) -> Tensor:
        guide = model.guide()
        self._set_last(model = model, y = y, x = x)
        dist = model(*x)
        log_density_dist = dist.log_density(y)
        log_density_guide = sum([guide_dist.log_density(guide_dist.sample()) for guide_dist in guide.values()])
        return LossTensor((loss_density_dist - loss_density_guide).data, self)
