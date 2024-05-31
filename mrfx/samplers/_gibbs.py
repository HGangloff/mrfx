"""
Classical Gibbs sampler
"""

import jax
import jax.numpy as jnp
from jaxtyping import Int, Float, Key, Array
import equinox as eqx
from mrfx.models._abstract import AbstractMarkovRandomFieldModel
from mrfx.samplers._abstract_gibbs import AbstractGibbsSampler
from mrfx.samplers._utils import get_neigh


class GibbsSampler(AbstractGibbsSampler):
    """ """

    lx: Int = eqx.field(static=True)
    ly: Int = eqx.field(static=True)
    eps: Float
    max_iter: Int

    def update_one_image(
        self,
        X: Array,
        model: AbstractMarkovRandomFieldModel,
        key: Key,
        key_permutation: Key,
        *_,
    ) -> Array:
        """
        Receives X at previous Gibbs iteration
        Outputs an updated X
        We do not need the last two arguments since no coloring happens in
        vanilla GibbsSampler
        """
        n_sites = self.lx * self.ly
        update_one_site_ = lambda carry, uv: self.update_one_site(*carry, model, uv)
        carry, _ = jax.lax.scan(
            update_one_site_,
            (X, key),
            jax.random.permutation(key_permutation, jnp.arange(n_sites)),
        )
        return carry[0]

    def update_one_site(
        self,
        X: Array,
        key: Key,
        model: AbstractMarkovRandomFieldModel,
        uv: Int,
    ) -> tuple[Array, Key]:
        u, v = jnp.unravel_index(uv, (self.lx, self.ly))
        neigh_values = get_neigh(X, u, v, self.lx, self.ly)
        key, subkey = jax.random.split(key, 2)
        potential_values = model.potential_values(neigh_values)
        x_sample = model.sample(potential_values, subkey)
        X = X.at[u, v].set(x_sample)
        return (X, key), None
