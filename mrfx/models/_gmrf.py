"""
Gaussian Markov Random Field model
"""

from jaxtyping import Float, Int
import equinox as eqx


class GMRF(eqx.Module):
    """
    Either (kappa, nu) are specified and the Matérn covariance function will be
    used. Or (r) is specified and the exponential covariance function is used.
    """

    kappa: Float = eqx.field(static=True, default=None, kw_only=True)
    nu: Float = eqx.field(static=True, default=None, kw_only=True)
    sigma: Float = eqx.field(static=True, default=None, kw_only=True)
    dim: Int = eqx.field(static=True, default=2, kw_only=True)
    r: Float = eqx.field(static=True, default=None, kw_only=True)

    def __post_init__(self):
        if self.sigma is None:
            # force a field with unit variance
            self.sigma = 1  # / jnp.sqrt(4 * jnp.pi * self.kappa ** 2)
        if self.nu is None:
            self.nu = 1.0

        if self.kappa is None and self.r is None:
            raise ValueError("(kappa, nu) or r must be specified")

        if self.kappa is not None and self.nu is not None and self.r is not None:
            raise ValueError("(kappa, nu) or r must be specified but not both")
