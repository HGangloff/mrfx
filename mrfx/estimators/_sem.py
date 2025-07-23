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

        def Stochastic_E_step(
            prior_model, condition_llkh_model, posterior_model, prev_X, key
        ):
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

            return sampler.run(posterior_model, key, prev_X)[1][-1], posterior_model

        def M_step(prior_model, condition_llkh_model, X, Y):
            prior_params = prior_model.estimate_parameters(
                X,
            )
            prior_model = prior_model.set_params(prior_params)
            condition_llkh_params = condition_llkh_model.estimate_parameters(Y, X)
            condition_llkh_model = condition_llkh_model.set_params(
                condition_llkh_params
            )
            return prior_model, condition_llkh_model

        # initialisation of a first set of parameters
        prior_model, condition_llkh_model = M_step(
            prior_model, condition_llkh_model, X_init, Y
        )

        def one_iteration(carry):
            prior_model, condition_llkh_model, posterior_model, X, key, i = carry
            jax.debug.print("SEM iteration {i}", i=i)

            # Stochastic E step
            key, subkey = jax.random.split(key, 2)
            X, posterior_model = Stochastic_E_step(
                prior_model, condition_llkh_model, posterior_model, X, subkey
            )
            # M step
            prior_model, condition_llkh_model = M_step(
                prior_model, condition_llkh_model, X, Y
            )
            i = i + 1
            return (prior_model, condition_llkh_model, posterior_model, X, key, i)

        i = 0
        carry = (prior_model, condition_llkh_model, posterior_model, X_init, key, i)
        prior_model, condition_llkh_model, posterior_model, _, _, i = (
            jax.lax.while_loop(
                lambda carry: carry[-1] + 1 <= self.max_iter, one_iteration, carry
            )
        )
        return prior_model, condition_llkh_model, posterior_model
