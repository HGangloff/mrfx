"""
Potts model
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, Key
import equinox as eqx

from mrfx.models._abstract import AbstractMarkovRandomFieldModel


class Potts(AbstractMarkovRandomFieldModel):
    """ """

    K: Int = eqx.field(static=True)
    beta: Float
    neigh_size: Int = eqx.field(kw_only=True, default=1, static=True) # default is 8 nearest neighbors

    def potential(self, x: Array, neigh_values: Array) -> Float:
        return self.beta * (x == neigh_values).sum()

    def potential_values(self, neigh_values: Array) -> Array:
        vmap_potential = jax.vmap(self.potential, (0, None))
        potential_values = jnp.exp(vmap_potential(jnp.arange(self.K), neigh_values))
        return potential_values / potential_values.sum()

    def sample(self, potential_values: Array, key: Key) -> Int:
        r = jax.random.uniform(key)
        potential_cum = jnp.cumsum(potential_values)
        return jnp.count_nonzero(potential_cum < r)
