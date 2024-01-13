from magic import Tensor
from magic import Module

def elbo(model: Module, y: Tensor, *x: Tensor) -> Tensor:
    guide = model.guide()
    dist = model(*x)
    log_density_dist = dist.log_density(y)
    log_desnity_guide = sum([guide_dist.log_density(guide_dist.sample()) for guide_dist in guide.values()])
    return loss_density_dist - loss_density_guide
