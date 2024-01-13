from magic.dist import Distribution

class Module:
    """ a magic module. used just like pytorch """

    def forward(self) -> None:
        raise NotImplementedError("must be implemented by subclass")

    def guide(self) -> None:
        raise NotImplementedError("must be implemented by subclass")

    def __repr__(self):
        attrs = [f"{key}={value}" for key, value in self.__dict__.items() if isinstance(value, Distribution)]
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def __call__(self, *args: object) -> None:
        return self.forward(*args)
