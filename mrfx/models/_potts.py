"""
Potts model
"""

from jaxtyping import Float, Array
import equinox as eqx

from mrfx.models._abstract import AbstractMarkovRandomFieldModel
from mrfx.abstract._distributions import AbstractPriorDistribution, Params


class PottsParameter(Params):
    beta: Float


class Potts(AbstractPriorDistribution, AbstractMarkovRandomFieldModel):
    """ """

    params: PottsParameter = eqx.field(kw_only=True)
    K: int = eqx.field(static=True, kw_only=True)
    neigh_size: int = eqx.field(
        kw_only=True, default=1, static=True
    )  # default is 8 nearest neighbors

    def potential(self, x: Array, neigh_values: Array, *args, **kwargs) -> Float:
        return self.params.beta * (x == neigh_values).sum(dtype=neigh_values.dtype)

    def estimate_parameters(self, self_realization: Array):
        raise NotImplementedError
