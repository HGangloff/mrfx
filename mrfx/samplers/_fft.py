"""
Fourier sampling of GMRF
"""

import jax
import jax.numpy as jnp
from jaxtyping import Int, Float, Key, Array
import equinox as eqx

from mrfx.models._gmrf import GMRF
from mrfx.samplers._utils import eval_matern_covariance, eval_exp_covariance


class FFTSamplerGMRF(eqx.Module):
    """
    The resulting GMRF will be on a torus
    """

    lx: Int = eqx.field(static=True, default=None, kw_only=True)
    ly: Int = eqx.field(static=True, default=None, kw_only=True)

    def get_base(self, model: GMRF) -> Float[Array, "lx ly"]:
        b = jnp.zeros((self.lx, self.ly))
        
        # TODO improve
        for x in range(self.lx):
            for y in range(self.ly):
                #b = b.at[x, y].set(eval_exp_covariance(model.sigma, 10, x=jnp.array([0., x]),
                #                                 y=jnp.array([0., y]),
                #                                 lx=self.lx, ly=self.ly))
                b = b.at[x, y].set(eval_matern_covariance(model.sigma, model.nu,
                                                 model.kappa, x=jnp.array([0., x]),
                                                 y=jnp.array([0., y]),
                                                 lx=self.lx, ly=self.ly))
        return b

    def get_base_invert_numerical(self, b: Float[Array, "lx ly"]) -> Float[Array, "lx ly"]:
        '''
        If b is the base of a matrix B, returns the base of B^-1 with the
        direct formula sing Fourier space
        '''
        B = jnp.fft.fft2(b, norm='ortho')
        #print(B)
        #mask = B.real > 1e-6
        #iB = jnp.zeros_like(B)
        #iB = iB.at[mask].set(jnp.power(B[mask], -1))
        #iB = iB.at[mask == 0].set(iB[mask].max())
        iB = jnp.power(B,-1)
        b_invert = 1 / (self.lx * self.ly) * jnp.real(jnp.fft.ifft2(iB, norm='ortho'))
        return b_invert

    def sample_image(self, model: GMRF, key: Key) -> Float[Array, "lx ly"]:
        """
        Sample image with Fourier sampling
        """
        base = self.get_base(model)
        base = base.at[0, 0].set(1.)
        base_invert = self.get_base_invert_numerical(base)
        key1, key2 = jax.random.split(key, 2)
        Z = (jax.random.normal(key1, shape=(self.lx, self.ly)) + 1j *
            jax.random.normal(key2, shape=(self.lx, self.ly)))
        return jnp.real(
            jnp.fft.fft2(
                jnp.power(
                    jnp.fft.fft2(base_invert),
                    -0.5
                ) * Z,
                norm="ortho"
            )
        )
