"""
Abstract MRF model
"""

import abc
from jaxtyping import Float, Int, Array, Key
import jax
import jax.numpy as jnp
import equinox as eqx

from mrfx.abstract._distributions import AbstractDistribution


class AbstractMarkovRandomFieldModel(AbstractDistribution):
    """
    Following the abstract final design Pattern, no init here, just
    eqx.AbstractVar"""

    K: eqx.AbstractVar[int]
    neigh_size: eqx.AbstractVar[int]

    @abc.abstractmethod
    def potential(self, x: Array, neigh_values: Array, u: Array, v: Array) -> Float:
        raise NotImplementedError

    def potential_values(self, neigh_values: Array, u: Array, v: Array) -> Array:
        vmap_potential = jax.vmap(self.potential, (0, None, None, None))
        potential_values = jnp.exp(
            vmap_potential(jnp.arange(self.K), neigh_values, u, v)
        )
        return potential_values / potential_values.sum()

    def sample(self, potential_values: Array, *, key: Key) -> Int:
        r = jax.random.uniform(key)
        potential_cum = jnp.cumsum(potential_values)
        return jnp.count_nonzero(potential_cum < r)
