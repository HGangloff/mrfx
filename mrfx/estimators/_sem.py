"""
Stochastic Expectation Maximization algorithm
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import fields

from jaxtyping import Array, Key
import jax
import equinox as eqx
from mrfx.abstract._iterative_algorithm import IterativeAlgorithm

if TYPE_CHECKING:
    from mrfx.abstract._sampler import AbstractSampler
    from mrfx.abstract._distributions import (
        AbstractPriorDistribution,
        AbstractConditionalLikelihoodDistribution,
        AbstractPosteriorDistribution,
    )


class StochasticExpectationMaximization(IterativeAlgorithm):
    """ """

    def run(
        self,
        prior_model: AbstractPriorDistribution,
        condition_llkh_model: AbstractConditionalLikelihoodDistribution,
        posterior_model: AbstractPosteriorDistribution,
        sampler: AbstractSampler,
        X_init: Array,
        Y: Array,
        *,
        key: Key,
    ):
        """ """

        def M_step(prior_model, condition_llkh_model, X, Y):
            prior_params = prior_model.estimate_parameters(
                X,
            )
            prior_model = prior_model.set_params(prior_params)
            condition_llkh_params = condition_llkh_model.estimate_parameters(X, Y)
            condition_llkh_model = condition_llkh_model.set_params(
                condition_llkh_params
            )
            return prior_model, condition_llkh_model

        # initialisation of a first set of parameters
        prior_model, condition_llkh_model = M_step(
            prior_model, condition_llkh_model, X_init, Y
        )

        X = X_init
        while True:
            # Stochastic E step
            # The order of the fields in all of the generated methods is the order in which they appear in the class definition.
            prior_params = tuple(
                getattr(prior_model.params, field.name)
                for field in fields(prior_model.params)
            )
            condition_llkh_params = tuple(
                getattr(condition_llkh_model.params, field.name)
                for field in fields(condition_llkh_model.params)
            )
            posterior_model = eqx.tree_at(
                lambda pt: tuple(
                    getattr(pt.params, field.name) for field in fields(pt.params)
                ),
                posterior_model,
                prior_params + condition_llkh_params,  # NOTE the ordering
            )

            key, subkey = jax.random.split(key, 2)
            sampler.run(posterior_model, subkey)

            # M step
            prior_model, condition_llkh_model = M_step(
                prior_model, condition_llkh_model, X, Y
            )
