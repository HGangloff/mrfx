"""
Spectral sampling of GMRF
"""

import jax
from jax import jit
import jax.numpy as jnp
from jaxtyping import Int, Float, Key, Array
import equinox as eqx

from mrfx.models._gmrf import GMRF
from mrfx.models._gum import GUM
from mrfx.samplers._utils import get_vertices


class SpectralSamplerGMRF(eqx.Module):
    """ """

    n_bands: Int = eqx.field(static=True, kw_only=True)
    lx: Int = eqx.field(static=True, default=None, kw_only=True)
    ly: Int = eqx.field(static=True, default=None, kw_only=True)

    def sample_one_loc(self, omega: Float[Array, "n_bands dim"], phi: Float[Array, "n_bands"], model: GMRF, x: Float[Array, "dim"]) -> Float:
        temp = jnp.exp(1j * (omega.T @ x + phi))
        return model.sigma * jnp.sqrt(2 / self.n_bands) * (
            jnp.sum(jnp.real(temp), axis=0)
        )

    def sample_omega_phi(self, model: GMRF, key: Key) -> tuple[Float[Array, "n_bands dim"], Float[Array, "n_bands"]]:
        """
        Spectral sampling getting rid of high frequencies
        """
        subkey = jax.random.split(key, 3)
        quantile = 0.95
        n_samples = int(self.n_bands * (1 + 2 * (1 - quantile)))
        gamma = jax.random.gamma(
            key=subkey[0],
            a=model.nu,
            shape=(n_samples,)
        ) * 2 / model.kappa ** 2
        xi = 1 / (2 * gamma)
        multR_plus = jnp.sqrt(2 * xi)
        threshold = jnp.quantile(multR_plus, quantile)
        #multR = multR_plus[multR_plus < threshold]
        omega = multR_plus[jnp.nonzero(multR_plus < threshold, size=self.n_bands)] * jax.random.normal(
            key=subkey[1],
            shape=(model.dim, self.n_bands)
        )
        phi = 2 * jnp.pi * jax.random.uniform(key=subkey[2], shape=(self.n_bands,))

        return omega, phi

    def sample_image(self, model: GMRF, key: Key) -> Float[Array, "lx ly"]:
        if model.dim != 2:
            raise ValueError("Cannot use sample_image for model whose dimension !=2")
        if self.lx is None or self.ly is None:
            raise ValueError("lx and ly must not be None to use sample_image")
        omega, phi = self.sample_omega_phi(model, key)
        stacked_sites = jnp.dstack(jnp.meshgrid(jnp.arange(self.lx),
            jnp.arange(self.ly))).reshape(-1, 2)
        v_sample_one_loc = jax.vmap(self.sample_one_loc, (None, None, None, 0))
        return v_sample_one_loc(omega, phi, model, stacked_sites).reshape((self.lx, self.ly))


class SpectralSamplerGUM(eqx.Module):
    """
    Sampling a GUM with internal spectral samplings for GMRF
    """

    n_bands: Int = eqx.field(static=True, kw_only=True)
    lx: Int = eqx.field(static=True, default=None, kw_only=True)
    ly: Int = eqx.field(static=True, default=None, kw_only=True)

    def sample_image(self, model: GUM, key: Key) -> Float[Array, "lx ly"]:
        if model.dim != 2:
            raise ValueError("Cannot use sample_image for model whose dimension !=2")
        if self.lx is None or self.ly is None:
            raise ValueError("lx and ly must not be None to use sample_image")
        gmrf = GMRF(kappa=model.kappa, dim=2) # dim is fixed since we want to
        gmrf_sampler = SpectralSamplerGMRF(n_bands=self.n_bands, lx=self.lx,
                ly=self.ly)
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
        return X

