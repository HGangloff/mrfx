"""
Gaussian Markov Random Field model
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
import equinox as eqx

class GMRF(eqx.Module):
    """ """

    kappa: Float = eqx.field(static=True, kw_only=True)
    nu: Float = eqx.field(static=True, default=1., kw_only=True)
    sigma: Float = eqx.field(static=True, default=None, kw_only=True)
    dim: Int = eqx.field(static=True, default=2, kw_only=True)

    def __post_init__(self):
        if self.sigma is None:
            # force a field with unit variance
            self.sigma = 1 / jnp.sqrt(4 * jnp.pi * self.kappa ** 2)
