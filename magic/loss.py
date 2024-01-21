import jax
import jax.numpy as np
from jax import grad, value_and_grad
from jax.scipy.stats import norm
from jax.experimental import optimizers

class ELBO:
    def __init__(self, model: object) -> None:
        self.model = model
        self.prior = model.guide()
        self.val = None

    def __call__(self, pred: np.array, y: np.array) -> np.array:
        params = self.model.get_params(self.model.opt_state)
        expected_log_likelihood = np.mean(norm.logpdf(y, pred, params["std"]))
        divergence = np.log(self.prior["std"] / params["std"]) + (params["std"] ** 2 + (params["mean"] - self.prior["mean"])**2) / (2 * self.prior["std"] ** 2) - 0.5
        elbo = expected_log_likelihood - divergence
        self.val = -elbo
        return self

    def backward(self):
        grad_fn = value_and_grad(lambda p: self(p, y)).val, argnums = 0)
        val, grad = grad_fn(self.model.params)
        self.model.update(grads)
        return loss_value
