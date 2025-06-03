"""
Abstract MRF model
"""

import abc
from jaxtyping import Float, Int, Array, Key
import equinox as eqx


class AbstractMarkovRandomFieldModel(eqx.Module):
    """ """

    K: Int = eqx.field(static=True)
    neigh_size: Int = eqx.field(
        kw_only=True, default=1, static=True
    )  # default is 8 nearest neighbors

    @abc.abstractmethod
    def potential(self, x: Array, neigh_values: Array) -> Float:
        raise NotImplementedError

    @abc.abstractmethod
    def potential_values(self, neigh_values: Array) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, potential_values: Array, key: Key) -> Array:
        raise NotImplementedError
