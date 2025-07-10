import pytest

import jax
import jax.numpy as jnp

from mrfx.samplers import ChromaticGibbsSampler
from mrfx.models import Potts


def test_ChromaticGibbsSampler():
    """
    Of course image size is ridiculous but this was a small and short run taken
    from a mrfx tag we believe to be correct. This will now become the main
    brick of reproducibility
    """
    key = jax.random.PRNGKey(0)
    K = 3
    beta = 1.0
    potts_model = Potts(K, beta=beta, neigh_size=1)

    key, subkey = jax.random.split(key, 2)
    lx = ly = 6
    chro_gibbs = ChromaticGibbsSampler(
        lx=lx, ly=ly, eps=0.05, max_iter=2, color_update_type="sequential_in_color"
    )
    X = chro_gibbs.run(potts_model, subkey)[1][-1]

    assert jnp.allclose(
        X,
        jnp.array(
            [
                [2, 2, 2, 2, 2, 1],
                [1, 1, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2],
                [1, 2, 2, 2, 2, 2],
                [2, 1, 1, 2, 2, 2],
                [2, 1, 2, 2, 2, 2],
            ],
            dtype=int,
        ),
    )
