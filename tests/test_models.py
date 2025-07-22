import pytest

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrfx.models import Potts

def test_potts_estimate_parameters():
    X = jnp.load("tests/potts_075.npy")
    potts = Potts(K=2, beta=0) # we do not know beta

    res = potts.estimate_parameters(X)
    plt.imshow(res)
    plt.show()
