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

def get_vertices(K):
    """ Return unitary simplex vertices coordinates for any dimension K-1
    
    Source : Anderson, G., & Thron, C. (2021). 
    Coordinate Permutation-Invariant Unit N-Simplexes in N dimensions. 
    Available at SSRN 3977222.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3977222
    """
    ######## Vertices coordinates
    # K = 3  # number of classes
    N = K-1 # dimension of the space

    vertices = jnp.zeros(shape=(K, N))
    ones = jnp.ones(shape=N)

    for j in range(N):
        # unit vector along dimension j
        e_j = jnp.zeros(shape=N)
        e_j = e_j.at[j].set(1.)
        
        v_j = jnp.sqrt((N + 1) / N) * e_j - 1 / (N * jnp.sqrt(N)) * (jnp.sqrt(N + 1) - 1) * ones
        vertices = vertices.at[j].set(v_j)

    vertices = vertices.at[N].set(-1 / jnp.sqrt(N) * ones)
    return vertices
