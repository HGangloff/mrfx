import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from mrfx.samplers import ChromaticGibbsSampler
from mrfx.models import AbstractMarkovRandomFieldModel
from mrfx.estimators import mpm_estimator
from mrfx.abstract import Params


def test_mpm():
    """
    Of course image size is ridiculous but this was a small and short run taken
    from a mrfx tag we believe to be correct. This will now become the main
    brick of reproducibility
    """
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key, 2)

    K = 3
    beta = 1.0
    lx = ly = 6
    X = jnp.array(
        [
            [2, 2, 2, 2, 2, 1],
            [1, 1, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2],
            [2, 1, 1, 2, 2, 2],
            [2, 1, 2, 2, 2, 2],
        ],
        dtype=int,
    )
    mu = jnp.arange(K)
    sigma = 0.5 * jnp.ones(K)
    Y = jax.random.normal(key, shape=X.shape) * jnp.sum(
        jnp.array([jnp.where(X == i, sigma[i], 0) for i in range(K)]), axis=0
    ) + jnp.sum(jnp.array([jnp.where(X == i, mu[i], 0) for i in range(K)]), axis=0)

    assert jnp.allclose(
        Y,
        jnp.array(
            [
                [2.502007, 1.5468314, 1.6259139, 1.4143165, 1.5643836, 1.294419],
                [1.361965, 0.48720092, 2.830814, 1.0544875, 1.3555331, 2.0668035],
                [1.4234804, 2.1196482, 2.8724036, 2.2525094, 2.2975266, 1.7635809],
                [1.4578187, 2.1806016, 1.6829178, 1.5647545, 2.379194, 2.5314267],
                [2.4541783, 0.7538399, 1.444246, 2.8226955, 0.7420671, 2.2208903],
                [2.9086564, 0.41343057, 1.7056417, 3.0734925, 2.4009402, 1.9012729],
            ],
            dtype=float,
        ),
    )

    class PosteriorFieldParameters(Params):
        beta: Float = eqx.field(kw_only=True)
        mu: Array = eqx.field(kw_only=True)
        sigma: Array = eqx.field(kw_only=True)

    class PosteriorField(AbstractMarkovRandomFieldModel):
        """ """

        params: PosteriorFieldParameters = eqx.field(kw_only=True)
        neigh_size: int = eqx.field(kw_only=True, static=True, default=1)
        K: int = eqx.field(kw_only=True, static=True)
        Y: Array = eqx.field(kw_only=True)

        def potential(self, x: Array, neigh_values: Array, u: Array, v: Array) -> Float:
            return (
                self.params.beta * (x == neigh_values).sum(dtype=neigh_values.dtype)
                - (self.Y[u, v] - self.params.mu[x]) ** 2
                / (2 * self.params.sigma[x] ** 2)
                - jnp.log(jnp.sqrt(2 * jnp.pi) * self.params.sigma[x])
            )

    params = PosteriorFieldParameters(beta=beta, mu=mu, sigma=sigma)
    post_field = PosteriorField(K=K, params=params, Y=Y)
    chro_gibbs = ChromaticGibbsSampler(
        lx=lx, ly=ly, eps=0.05, max_iter=2, color_update_type="sequential_in_color"
    )

    key, subkey = jax.random.split(key, 2)
    X_MPM = mpm_estimator(chro_gibbs, post_field, K, 15, subkey)

    assert jnp.allclose(
        X_MPM,
        jnp.array(
            [
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [2, 1, 2, 2, 2, 2],
            ],
            dtype=int,
        ),
    )
