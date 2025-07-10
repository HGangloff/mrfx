import abc
from typing import Any
from jax import jit
import equinox as eqx


class AbstractSampler(eqx.Module):
    """ """

    lx: int = eqx.field(kw_only=True, static=True)
    ly: int = eqx.field(kw_only=True, static=True)

    @abc.abstractmethod
    def run(self, *_, **__) -> Any:
        raise NotImplementedError

    @jit
    @abc.abstractmethod
    def update_one_image(self, *_, **__) -> Any:
        raise NotImplementedError
