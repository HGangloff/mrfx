"""
Some utility functions
"""

import jax
import jax.numpy as jnp
import scipy
import numpy as np
import scipy.special
from jax import custom_jvp, pure_callback, vmap


def get_neigh(X, u, v, lx, ly, neigh_size):
    if neigh_size == 1:
        # Explicitly avoid padding for this simple neighborhood for speed
        # efficiency
        top = X[(u + 1) % ly, v]
        bot = X[(u - 1) % ly, v]

        t_r = X[(u + 1) % ly, (v + 1) % lx]
        t_l = X[(u + 1) % ly, (v - 1) % lx]
        b_r = X[(u - 1) % ly, (v + 1) % lx]
        b_l = X[(u - 1) % ly, (v - 1) % lx]

        right = X[u, (v + 1) % lx]
        left = X[u, (v - 1) % lx]

        return jnp.array([left, t_l, top, t_r, right, b_r, bot, b_l])
    # mode='wrap' does not help convergence as opposed to mode="constant"
    # Mandatory to pad everytime because a jax.lax.cond could not output
    # varying shapes
    X_padded = jnp.pad(X, pad_width=neigh_size, mode='wrap')
    slice_ = jax.lax.dynamic_slice(
        X_padded,
        (u - neigh_size, v - neigh_size),
        (2 * neigh_size + 1, 2 * neigh_size + 1)
    )
    return jnp.delete(slice_, (2 * neigh_size + 1) * neigh_size + neigh_size,
                    axis=None, assume_unique_indices=True) # axis=None -> operate on flattened array

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

def euclidean_dist(x1, x2, y1, y2, *args, **kwargs):
    '''
    '''
    return jnp.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def euclidean_dist_torus(x1, x2, y1, y2, lx, ly):
    '''
    '''
    return jnp.sqrt(jnp.min(jnp.array([abs(x1 - x2), lx - abs(x1 - x2)])) ** 2 +
                   jnp.min(jnp.array([abs(y1 - y2), ly - abs(y1 - y2)])) ** 2)

def eval_matern_covariance(sigma, nu, kappa, x1=None, x2=None, y1=None, y2=None, h=None, lx=None, ly=None):
    """ If lx and ly are not None, this is matern distance is computed on the
    torus. Specify either x and y the two points or their distance h"""
    if (x1 is None or y1 is None) and h is None:
        raise ValueError("(x,y) or h must be specified")
    if (x1 is not None and y1 is not None) and h is not None:
        raise ValueError("(x,y) and h cannot be specified together")
    if lx is not None and ly is not None:
        sc_h = kappa * euclidean_dist_torus(x1, x2, y1, y2, lx, ly)
    else:
        if h is None:
            sc_h = kappa * jnp.sum((x1 - x2) ** 2 + (y1 - y2) ** 2, axis=-1)
        else:
            sc_h = kappa * jnp.linalg.norm(h, axis=-1)
    return sigma ** 2 / jnp.exp(jax.scipy.special.gammaln(nu)) / (2
            ** (nu - 1)) * ((sc_h) ** nu) * kv(nu, sc_h)

def eval_exp_covariance(sigma, r, x1=None, x2=None, y1=None, y2=None, h=None, lx=None, ly=None):
    """ If lx and ly are not None, this is matern distance is computed on the
    torus. Specify either x and y the two points or their distance h"""
    if (x1 is None or y1 is None) and h is None:
        raise ValueError("(x,y) or h must be specified")
    if (x1 is not None and y1 is not None) and h is not None:
        raise ValueError("(x,y) and h cannot be specified together")
    if lx is not None and ly is not None:
        h = euclidean_dist_torus(x1, x2, y1, y2, lx, ly)
    else:
        if h is None:
            h = jnp.sum((x1 - x2) ** 2 + (y1 - y2) ** 2, axis=-1)
        else:
            h = jnp.linalg.norm(h, axis=-1)
    res = sigma ** 2 * jnp.exp(- h ** 2 / r)
    #jax.debug.print("{x}", x=res)
    return jax.lax.cond(res < 1e-6, lambda _: 0., lambda _: res, (None,))

def generate_modified_bessel(function, sign):
    """function is Kv and Iv"""

    @custom_jvp
    def cv(v, x):
        return pure_callback(
            lambda vx: function(*vx),
            x,
            (v, x),
            vectorized=True,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.cond(
            v == 0,
            lambda: sign * cv(v + 1, x),
            lambda: 0.5 * (cv(v - 1, x) + cv(v + 1, x)),
        )

        return primal_out, tangents_out * dx

    return cv


kv = generate_modified_bessel(scipy.special.kv, sign=-1)
