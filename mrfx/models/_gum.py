"""
Gaussian Unit-simplex Markov Random Field model
"""

from jaxtyping import Int
import equinox as eqx

from mrfx.models import GMRF


class GUM(GMRF):
    """ """

    K: Int = eqx.field(static=True, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
