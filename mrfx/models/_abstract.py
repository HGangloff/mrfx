"""
Abstract MRF model
"""

import abc
from jaxtyping import Float, Int, Array, Key
import equinox as eqx


class AbstractMarkovRandomFieldModel(eqx.Module):
    """ """

    K: eqx.AbstractVar[Int]

    @abc.abstractmethod
    def potential(self, x: Array, neigh_values: Array) -> Float:
        raise NotImplementedError

    @abc.abstractmethod
    def potential_values(self, neigh_values: Array) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, potential_values: Array, key: Key) -> Array:
        raise NotImplementedError
