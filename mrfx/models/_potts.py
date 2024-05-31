"""
Potts model
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array
import equinox as eqx

from mrfx.models._abstract import AbstractMarkovRandomFieldModel


class Potts(AbstractMarkovRandomFieldModel):
    """ """

    K: Int = eqx.field(static=True)
    beta: Float

    def potential(self, x: Array, neigh_values: Array) -> Float:
        return self.beta * (x == neigh_values).sum()

    def potential_values(self, neigh_values: Array) -> Array:
        vmap_potential = jax.vmap(self.potential, (0, None))
        potential_values = jnp.exp(vmap_potential(jnp.arange(self.K), neigh_values))
        return potential_values / potential_values.sum()
