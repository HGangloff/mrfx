"""
Some utility functions
"""

import jax
import jax.numpy as jnp
import scipy


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


def euclidean_dist_torus(x, y, lx, ly):
    """ Euclidean distance on the torus """
    return jnp.sqrt(min(jnp.abs(x[0] - x[1]), lx - jnp.abs(x[0] - x[1])) ** 2 +
                   min(jnp.abs(y[0] - y[1]), ly - jnp.abs(y[0] - y[1])) ** 2)

def eval_matern_covariance(sigma, nu, kappa, x=None, y=None, h=None, lx=None, ly=None):
    """ If lx and ly are not None, this is matern distance is computed on the
    torus. Specify either x and y the two points or their distance h"""
    if (x is None or y is None) and h is None:
        raise ValueError("(x,y) or h must be specified")
    if (x is not None and y is not None) and h is not None:
        raise ValueError("(x,y) and h cannot be specified together")
    if lx is not None and ly is not None:
        sc_h = kappa * euclidean_dist_torus(x, y, lx, ly)
    else:
        if h is None:
            sc_h = kappa * jnp.linalg.norm(x - y, axis=-1)
        else:
            sc_h = kappa * jnp.linalg.norm(h, axis=-1)
    return sigma ** 2 / jnp.exp(jax.scipy.special.gammaln(nu)) / (2
            ** (nu - 1)) * ((sc_h) ** nu) * scipy.special.kv(nu, sc_h)

def eval_exp_covariance(sigma, r, x=None, y=None, h=None, lx=None, ly=None):
    """ If lx and ly are not None, this is matern distance is computed on the
    torus. Specify either x and y the two points or their distance h"""
    if (x is None or y is None) and h is None:
        raise ValueError("(x,y) or h must be specified")
    if (x is not None and y is not None) and h is not None:
        raise ValueError("(x,y) and h cannot be specified together")
    if lx is not None and ly is not None:
        h =  euclidean_dist_torus(x, y, lx, ly)
    else:
        if h is None:
            h = jnp.linalg.norm(x - y, axis=-1)
        else:
            h = jnp.linalg.norm(h, axis=-1)
    return sigma ** 2 * jnp.exp(- h / r)

