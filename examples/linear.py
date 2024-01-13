import magic
import magic.dist as dist

class Model(magic.Module):
    """ bayesian linear regression """

    def __init__(self) -> None:
        self.a = dist.Normal()
        self.b_a = dist.Normal()
        self.b_r = dist.Normal()
        self.b_ar = dist.Normal()
        self.sigma = dist.Normal()

        self.a_loc = 0.0
        self.a_scale = 1.0
        self.sigma_loc = 1.0 
        self.weights_loc = [0.0, 0.0, 0.0]
        self.weights_scale = [1.0, 1.0, 1.0] 

    def forward(self, is_cont_africa: int, ruggedness: float) -> dist.Normal:
        mean = self.a + self.b_a * is_cont_africa + self.b_r * ruggedness + self.b_ar * is_cont_africa * ruggedness
        return dist.Normal(mean, self.sigma)

    def guide(self) -> dist.Normal:
        return {
            "a": dist.Normal(self.a_loc, self.a_scale),
            "b_a": dist.Normal(self.weights_loc[0], self.weights_loc[0]),
            "b_r": dist.Normal(self.weights_loc[1], self.weights_loc[1]),
            "b_ar": dist.Normal(self.weights_loc[2], self.weights_loc[1]),
            "sigma": dist.Normal(self.sigma_loc, 0.05)
        }

m = Model()
print(m)
