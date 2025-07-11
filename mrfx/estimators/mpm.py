from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import equinox as eqx
from jaxtyping import Int, Array, Key

from mrfx.utils.utils import get_most_frequent_values

if TYPE_CHECKING:
    from mrfx.samplers import AbstractSampler
    from mrfx.models import AbstractMarkovRandomFieldModel


def mpm_estimator(
    sampler: AbstractSampler,
    model: AbstractMarkovRandomFieldModel,
    K: int,
    n: int,
    key: Key,
) -> Int[Array]:
    """
    Get the maximum posterior mode estimate of a MRF
    """
    # Silence the sampler
    sampler = eqx.tree_at(lambda pt: pt.verbose, sampler, False)

    def one_simulation(key):
        return sampler.run(model, key)[1][-1]

    keys = jax.random.split(key, n)
    samples = jax.vmap(one_simulation)(keys)
    return get_most_frequent_values(samples, K)
