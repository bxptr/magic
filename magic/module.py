from magic.dist import Distribution

import itertools

class Module:
    """ a magic module"""

    def forward(self) -> None:
        raise NotImplementedError("must be implemented by subclass")

    def guide(self) -> None:
        raise NotImplementedError("must be implemented by subclass")

    def params(self) -> dict:
        params = []
        def _get_params(dist: Distribution) -> None:
            for p in dist.params:
                if isinstance(p, Distribution):
                    _get_params(p)
                else:
                    params.append(p)
        for dist in self.__dict__.values():
            _get_params(dist)
        return params

    def _jax_forward(self, *args: object) -> None:
        return self.forward(*args).sample().data[0]

    def __repr__(self):
        attrs = [f"{key}={value}" for key, value in self.__dict__.items() if isinstance(value, Distribution)]
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def __call__(self, *args: object) -> None:
        return self.forward(*args)
