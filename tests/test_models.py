import jax.numpy as jnp

from mrfx.models import Potts


def test_potts_estimate_parameters():
    X = jnp.load("tests/potts_075.npy")
    potts = Potts(K=2, beta=0)  # we do not know beta

    res = potts.estimate_parameters(X)
    assert jnp.allclose(res, 2.57, atol=1e-2)
