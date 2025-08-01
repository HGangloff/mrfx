import jax.numpy as jnp

from mrfx.models import Potts


def test_potts_estimate_parameters_derin():
    X = jnp.load("tests/potts_075.npy")
    potts = Potts(K=2, beta=0)  # we do not know beta

    res = potts.estimate_parameters(X, method="derin")
    assert jnp.allclose(res.beta, 2.57, atol=1e-2)


def test_potts_estimate_parameters_farag():
    X = jnp.load("tests/potts_075.npy")
    potts = Potts(K=2, beta=0)  # we do not know beta

    res = potts.estimate_parameters(X, method="farag")
    assert jnp.allclose(res.beta, 1.78, atol=1e-2)
