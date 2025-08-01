"""
Gaussian Iid model
"""

from __future__ import annotations

from dataclasses import InitVar
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

    params: GaussianIidParameter = eqx.field(kw_only=True, default=None)
    prior_model: AbstractPriorDistribution = eqx.field(kw_only=True, static=True)
    mu: InitVar[float] = eqx.field(default=None)
    sigma: InitVar[float] = eqx.field(default=None)

    def __post_init__(self, mu=None, sigma=None):
        """
        We let the user specify the parameter either via params or (mu and
        sigma), the latter being less verbose
        """
        if self.params is None and (mu is None or sigma is None):
            raise ValueError("`params` or (`mu` and `sigma`) must be specified")
        if self.params is not None and (mu is not None or sigma is not None):
            raise ValueError(
                "`params` or (`mu` or `sigma`) cannot be specified together"
            )
        if self.params is None and (mu is not None and sigma is not None):
            self.params = GaussianIidParameter(mu=mu, sigma=sigma)

    def sample(self, prior_realization: Array, key: Key) -> Array:
        return jax.random.normal(key, shape=prior_realization.shape) * jnp.sum(
            jnp.array(
                [
                    jnp.where(prior_realization == i, self.params.sigma[i], 0)
                    for i in range(self.prior_model.K)
                ]
            ),
            axis=0,
        ) + jnp.sum(
            jnp.array(
                [
                    jnp.where(prior_realization == i, self.params.mu[i], 0)
                    for i in range(self.prior_model.K)
                ]
            ),
            axis=0,
        )

    def evaluate_pdf(self, realization: Array, prior_realization: Array) -> Array:
        """
        Evaluate p(y|prior_realization)
        """
        raise NotImplementedError
        # return jax.scipy.stats.norm.pdf(
        #    realization, loc=self.params.mu, scale=self.params.sigma
        # )

    def estimate_parameters(
        self, self_realization: Array, prior_realization: Array
    ) -> Params:
        """
        Order of returned arguments must match definition above because of
        `params.setter`

        **Note** that because of shapes are fixed in jitted code, the jnp.where
        has fixed size and we need to fill with NaN for the locations that do
        not belong to a particular prior_realization class. Afterwards, those
        NaN values are excluded from the estimator computations.
        """

        mu = jnp.array(
            [
                jnp.nanmean(
                    jnp.where(
                        prior_realization == i,
                        self_realization,
                        jnp.nan,
                    )
                )
                for i in range(self.prior_model.K)
            ]
        )
        sigma = jnp.array(
            [
                jnp.nanstd(
                    jnp.where(
                        prior_realization == i,
                        self_realization,
                        jnp.nan,
                    )
                )
                for i in range(self.prior_model.K)
            ]
        )
        return GaussianIidParameter(mu=mu, sigma=sigma)
