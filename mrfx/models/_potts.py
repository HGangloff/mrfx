"""
Potts model
"""

from jaxtyping import Float, Array
import equinox as eqx

from mrfx.models._abstract import AbstractMarkovRandomFieldModel


class Potts(AbstractMarkovRandomFieldModel):
    """ """

    beta: Float = eqx.field(kw_only=True)

    def potential(self, x: Array, neigh_values: Array, *args, **kwargs) -> Float:
        return self.beta * (x == neigh_values).sum(dtype=neigh_values.dtype)
