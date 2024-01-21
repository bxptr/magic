import jax
import jax.numpy as np

class Normal:
    """ normal distribution """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.params = {"mean": np.array(mean), "std": np.array(std)}
        self.key = jax.random.PRNGKey(0)

    def log_density(self, x: np.array) -> np.array:
        const = np.log(np.sqrt(2 * np.pi))
        sigma = self.std ** 2
        return (-((x - self.mean) ** 2) / (sigma * 2) - np.log1p(self.std.data) - const)[0]

    def sample(self, shape: tuple = (1,)) -> np.array:
        self.key, subkey = jax.random.split(self.key)
        return self.mean + self.std * jax.random.normal(subkey, shape)
