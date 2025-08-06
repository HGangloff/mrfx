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


class GaussianParameter(Params):
    mu: Array = eqx.field(kw_only=True)
    cov: Array = eqx.field(kw_only=True)


class Gaussian(AbstractConditionalLikelihoodDistribution):
    """ """

    dim: int = eqx.field(
        static=True,
    )
    prior_model: AbstractPriorDistribution = eqx.field(kw_only=True, static=True)
    params: GaussianParameter = eqx.field(kw_only=True, default=None)
    mu: InitVar[float] = eqx.field(default=None)
    cov: InitVar[float] = eqx.field(default=None)

    def __post_init__(self, mu=None, cov=None):
        """
        We let the user specify the parameter either via params or (mu and
        cov), the latter being less verbose
        """
        if self.params is None and (mu is None or cov is None):
            raise ValueError("`params` or (`mu` and `cov`) must be specified")
        if self.params is not None and (mu is not None or cov is not None):
            raise ValueError("`params` or (`mu` or `cov`) cannot be specified together")
        if self.params is None and (mu is not None and cov is not None):
            self.params = GaussianParameter(mu=mu, cov=cov)

    def mu_x(self, prior_realization: Array) -> Array:
        """
        Get mu conditionally to a prior realization
        """
        return jnp.sum(
            jnp.array(
                [
                    jnp.where(prior_realization == i, self.params.mu[i], 0)
                    for i in range(self.prior_model.K)
                ]
            ),
            axis=0,
        )

    def sample(self, prior_realization: Array, key: Key) -> Array:
        return jax.random.multivariate_normal(
            key,
            self.mu_x(prior_realization),
            self.params.cov,
            shape=prior_realization.shape,
        )

    def evaluate_pdf(self, realization: Array, prior_realization: Array) -> Array:
        """
        Evaluate p(y|prior_realization)
        """
        return jax.scipy.stats.multivariate_normal.pdf(
            realization, self.mu_x(prior_realization), self.params.cov
        )

    def evaluate_logpdf(self, realization: Array, prior_realization: Array) -> Array:
        """
        Evaluate log p(y|prior_realization)
        """
        return jax.scipy.stats.multivariate_normal.logpdf(
            realization, self.mu_x(prior_realization), self.params.cov
        )

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
        values_by_K = jnp.array(
            [
                jnp.where(
                    prior_realization[..., None] == i,
                    self_realization,
                    jnp.nan,
                )
                for i in range(self.prior_model.K)
            ]
        )

        mu = jnp.nanmean(values_by_K, axis=(1, 2))
        # One cov for all K
        cov = jnp.cov(
            self_realization.reshape((-1, self.dim)),
            rowvar=False,
        )
        # One cov for each K
        # cov = jnp.array(
        #    [
        #        jnp.cov(
        #            values_by_K[i].reshape((-1, self.dim)),
        #            rowvar=False,
        #            aweights=jnp.where(
        #                jnp.isnan(
        #                    jnp.sum(
        #                        values_by_K[i].reshape((-1, self.dim)),
        #                        axis=-1
        #                    )
        #                ),
        #                0,
        #                1
        #            )
        #        )
        #        for i in range(self.prior_model.K)
        #    ]
        # )
        return GaussianParameter(mu=mu, cov=cov)
