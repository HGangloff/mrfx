import jax.numpy as jnp
from mrfx.models._GaussianIid import GaussianIid, GaussianIidParameter
from mrfx.models._potts import Potts, PottsParameter


def test_set_parameters():
    cond_llkh = GaussianIid(
        params=GaussianIidParameter(mu=2 * jnp.ones((4,)), sigma=jnp.ones((4,))),
        prior_model=Potts(K=2, neigh_size=1, params=PottsParameter(beta=0.2)),
    )

    assert jnp.allclose(cond_llkh.params.mu, jnp.ones((4,)) * 2)
    assert jnp.allclose(cond_llkh.params.sigma, jnp.ones((4,)))

    cond_llkh = cond_llkh.set_params(
        GaussianIidParameter(mu=4 * jnp.ones((4,)), sigma=3 * jnp.ones((4,)))
    )
    assert jnp.allclose(cond_llkh.params.mu, jnp.ones((4,)) * 4)
    assert jnp.allclose(cond_llkh.params.sigma, jnp.ones((4,)) * 3)
