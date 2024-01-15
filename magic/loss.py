from magic import Tensor
from magic import Module

import jax
import jax.numpy as np

from typing import Union
from magic.dist import Distribution

class LossTensor(Tensor):
    def __init__(self, data: object, loss: "Loss") -> None:
        super().__init__(data)
        self._loss = loss

    def backward(self) -> None:
        print("Debugging - Last inputs:", self._loss._last)  # Debugging statement

        grad_fn = jax.grad(self._loss.model._jax_forward, argnums=0, allow_int=True)
        gradients = grad_fn(*self._loss._last)

        print("Debugging - Gradients:", gradients)  # Debugging statement

        # Apply the gradients to update the model parameters
        # self._loss.model.apply_grad(gradients)

class Loss:
    """ base loss class """

    def __init__(self) -> None:
        self._last = []

    def __call__(self, *args: object) -> Tensor:
        raise NotImplementedError("must be implemented by subclass")

class ELBO(Loss):
    """ ELBO loss """

    def __init__(self, model: Module) -> None:
        super().__init__()
        self.model = model

    def __call__(self, dist: Tensor, y: Union[Tensor, float], x: Union[Tensor, float]) -> Tensor:
        guide = self.model.guide()
        if type(x) == Tensor:
            self._last = [x.data]
        else:
            self._last = [x]
        log_density_dist = dist.log_density(y)
        log_density_guide = sum([guide_dist.log_density(guide_dist.sample()) for guide_dist in guide.values()])
        return LossTensor((log_density_dist - log_density_guide).data, self)
