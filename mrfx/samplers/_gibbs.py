"""
Classical Gibbs sampler
"""

import jax
from jax import jit
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Int, Key, Array

from mrfx.models._abstract_mrf import AbstractMarkovRandomFieldModel
from mrfx.samplers._abstract_gibbs import AbstractGibbsSampler
from mrfx.samplers._utils import get_neigh


class GibbsSampler(AbstractGibbsSampler):
    """ """

    name: str = eqx.field(static=True, kw_only=True, default="Gibbs sampler")

    @jit
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

    @jit
    def update_one_site(
        self,
        X: Array,
        key: Key,
        model: AbstractMarkovRandomFieldModel,
        uv: Int,
    ) -> tuple[tuple[Array, Key], None]:
        u, v = jnp.unravel_index(uv, (self.lx, self.ly))
        neigh_values = get_neigh(X, u, v, self.lx, self.ly, model.neigh_size)
        key, subkey = jax.random.split(key, 2)
        potential_values = model.potential_values(neigh_values, u, v)
        x_sample = model.sample(potential_values, key=subkey)
        X = X.at[u, v].set(x_sample)
        return (X, key), None
