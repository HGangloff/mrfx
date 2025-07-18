"""
Stochastic Expectation Maximization algorithm
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from jaxtyping import Array, Key

from mrfx.abstract._iterative_algorithm import IterativeAlgorithm

if TYPE_CHECKING:
    from mrfx.abstract._sampler import AbstractSampler
    from mrfx.abstract._distributions import AbstractJointDistribution


class StochasticExpectationMaximization(IterativeAlgorithm):
    """ """

    def run(
        self,
        model: AbstractJointDistribution,
        sampler: AbstractSampler,
        X_init: Array,
        key: Key,
    ):
        """ """

        while True:
            model.estimate_parameters(None)
