"""
Gaussian Iid model
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from jaxtyping import Key, Array
import equinox as eqx
import jax
import jax.numpy as jnp

from mrfx.abstract._distributions import (
    AbstractConditionalLikelihoodDistribution,
    Params,
)

if TYPE_CHECKING:
    from mrfx.abstract._distributions import AbstractPriorDistribution


class GaussianIidParameter(Params):
    mu: Array = eqx.field(kw_only=True)
    sigma: Array = eqx.field(kw_only=True)


class GaussianIid(AbstractConditionalLikelihoodDistribution):
    """ """

    params: GaussianIidParameter = eqx.field(kw_only=True)
    prior_model: AbstractPriorDistribution = eqx.field(kw_only=True, static=True)

    @property
    def mu(self):
        return self.params.mu

    @property
    def sigma(self):
        return self.params.sigma

    def sample(self, prior_realization: Array, key: Key) -> Array:
        return jax.random.normal(key, shape=prior_realization.shape) * jnp.sum(
            jnp.array(
                [
                    jnp.where(prior_realization == i, self.sigma[i], 0)
                    for i in range(self.prior_model.K)
                ]
            ),
            axis=0,
        ) + jnp.sum(
            jnp.array(
                [
                    jnp.where(prior_realization == i, self.mu[i], 0)
                    for i in range(self.prior_model.K)
                ]
            ),
            axis=0,
        )

    def evaluate_pdf(self, realization: Array, prior_realization: Array) -> Array:
        """
        Evaluate p(y|prior_realization)
        """
        return jax.scipy.stats.norm.pdf(
            realization,
            loc=self.params.mu,
            scale=self.params.sigma
        )

    def estimate_parameters(
        self, self_realization: Array, prior_realization: Array
    ) -> Params:
        """
        Order of returned arguments must match definition above because of
        `params.setter`
        """

        mu = jnp.array(
            [
                jnp.nanmean(
                    prior_realization[
                        jnp.nonzero(
                            prior_realization == i,
                            size=prior_realization.size,
                            fill_value=jnp.nan,
                        )
                    ]
                )
                for i in range(self.prior_model.K)
            ]
        )
        sigma = jnp.array(
            [
                jnp.nanstd(
                    prior_realization[
                        jnp.nonzero(
                            prior_realization == i,
                            size=prior_realization.size,
                            fill_value=jnp.nan,
                        )
                    ]
                )
                for i in range(self.prior_model.K)
            ]
        )
        return (mu, sigma)
