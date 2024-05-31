"""
Some utility functions
"""

import jax
import jax.numpy as jnp


def get_neigh(X, u, v, lx, ly):
    top = X[(u + 1) % ly, v]
    bot = X[(u - 1) % ly, v]

    t_r = X[(u + 1) % ly, (v + 1) % lx]
    t_l = X[(u + 1) % ly, (v - 1) % lx]
    b_r = X[(u - 1) % ly, (v + 1) % lx]
    b_l = X[(u - 1) % ly, (v - 1) % lx]

    right = X[u, (v + 1) % lx]
    left = X[u, (v - 1) % lx]

    return jnp.array([left, t_l, top, t_r, right, b_r, bot, b_l])
