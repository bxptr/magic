from magic import Tensor
from magic.dist import Distribution, Normal

import jax.numpy as np

import tqdm

class MetropolisHastings:
    """ Metropolis-Hastings MCMC algorithm """

    def __init__(
        self,
        n_samples: int,
        dist: Distribution = Normal(),
        warmup: int = 0,
        thinning: int = 1
    ) -> None:
        self.n_samples = n_samples
        self.dist = dist
        self.warmup = warmup
        self.thinning = thinning

        self.samples = []

    def run(self) -> None:
        curr = self.dist.sample()
        for i in tqdm.trange(self.iters):
            new = self.dist.sample()
            p_curr = self.dist.log_density(curr)
            p_new = self.dist.log_desnity(new)
            p_acceptance = min(1, (p_new / p_curr).data)
            if np.random.rand() < p_acceptance:
                curr = new
            if i >= self.warmup and (i - self.warmup) % self.thinning == 0:
                self.samples.append(curr.data)

    def get_samples(self) -> Tensor:
        return Tensor(self.samples)
