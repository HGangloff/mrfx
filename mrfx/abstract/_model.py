import abc
from typing import Any
from jaxtyping import Key
import equinox as eqx


class AbstractModel(eqx.Module):
    """ """

    @abc.abstractmethod
    def run(self, *_, key: Key, **__) -> Any:
        raise NotImplementedError
