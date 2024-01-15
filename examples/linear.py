import magic
from magic import Tensor
import magic.dist as dist
import magic.loss as loss

class Model(magic.Module):
    """ bayesian linear regression """

    def __init__(self) -> None:
        self.alpha = dist.Normal(0, 10)
        self.beta = dist.Normal(0, 10)
        self.sigma = dist.Normal(0, 10)

    def forward(self, x: Tensor) -> dist.Normal:
        mean = self.alpha + self.beta * x
        return dist.Normal(mean, self.sigma)

    def guide(self) -> dist.Normal:
        alpha_loc = Tensor(0.)
        alpha_scale = Tensor(1.)
        beta_loc = Tensor(0.)
        beta_scale = Tensor(1.)
        sigma_loc = Tensor(0.)
        return {
            "alpha": dist.Normal(alpha_loc, alpha_scale),
            "beta": dist.Normal(beta_loc, beta_scale),
            "sigma": dist.Normal(sigma_loc, Tensor(0.05))
        }

m = Model()
l = loss.ELBO(m)

x = Tensor(0)
y = Tensor(1)
pred = m(x)
a = l(pred, y, x)
print("-", a)
print("-", a.backward())
