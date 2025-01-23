"""
Fourier sampling or spectral sampling of GUM
"""

import jax
import jax.numpy as jnp
from typing import Literal
from jaxtyping import Int, Float, Key, Array
import equinox as eqx

from mrfx.models._gmrf import GMRF
from mrfx.models._gum import GUM

from mrfx.samplers._spectral import SpectralSamplerGMRF
from mrfx.samplers._fft import FFTSamplerGMRF
from mrfx.samplers._utils import get_vertices

class GUMSampler(eqx.Module):
    """
    Sampling a GUM
    """
    method: Literal["spectral", "fourier"] = eqx.field(static=True,
                                                       kw_only=True,
                                                       default="spectral")
    n_bands: Int = eqx.field(static=True, kw_only=True, default=None)
    lx: Int = eqx.field(static=True, default=None, kw_only=True)
    ly: Int = eqx.field(static=True, default=None, kw_only=True)

    def sample_image(self, model: GUM, key: Key) -> Float[Array, "lx ly"]:
        if model.dim != 2:
            raise ValueError("Cannot use sample_image for model whose dimension !=2")
        if self.lx is None or self.ly is None:
            raise ValueError("lx and ly must not be None to use sample_image")
        gmrf = GMRF(kappa=model.kappa, dim=2) # dim is fixed since we want to
        if self.method == "spectral":
            if self.n_bands is None:
                raise ValueError("n_bands must be provided for spectral sampling")
            gmrf_sampler = SpectralSamplerGMRF(n_bands=self.n_bands, lx=self.lx, ly=self.ly)
        elif self.method == "fourier":
            gmrf_sampler = FFTSamplerGMRF(lx=self.lx, ly=self.ly)
        else:
            raise ValueError("Wrong method argument")
        subkeys = jax.random.split(key, model.K - 1)
        z = jnp.stack([gmrf_sampler.sample_image(gmrf, subkeys[k]) for k in
            range(model.K - 1)], axis=0)
        vertices = get_vertices(model.K)
        z_fl = z.reshape(model.K - 1, self.lx * self.ly)

        dists = jnp.stack([
            jnp.linalg.norm(z_fl - vertices[v].reshape((-1, 1)),axis=0) for v in
                range(model.K)], axis=0)
 
        closest = jnp.argmin(dists, axis=0)
        X = closest.reshape(self.lx, self.ly)
        return X, jnp.min(dists, axis=0).reshape(self.lx, self.ly)
