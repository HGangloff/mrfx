import jax
import jax.numpy as jnp

from mrfx.models import GUM
from mrfx.samplers import GUMSampler

from mrfx.models import Potts
from mrfx.samplers import ChromaticGibbsSampler
from mrfx.experiments import time_complete_sampling

key = jax.random.PRNGKey(0)
kappa = 0.1

sizes = [(2**e, 2**e) for e in range(4, 10)]
reps = 1000
kappa = 0.1


def init_callback_fun(key, K, lx, ly):
    gum = GUM(kappa=kappa, K=K, dim=2)
    gum_sampler = GUMSampler(n_bands=2500, lx=lx, ly=ly, method="spectral")
    return gum_sampler.sample_image(gum, subkey)[0]


key, subkey = jax.random.split(key, 2)
Ks = jnp.arange(2, 3)

# K = 2
times, n_iterations, _ = time_complete_sampling(
    Sampler=ChromaticGibbsSampler,
    Model=Potts,
    key=subkey,
    Ks=Ks,
    sizes=sizes,
    reps=reps,
    kwargs_sampler={
        "eps": 0.05,
        "max_iter": 1000,
        "color_update_type": "sequential_in_color",
        "verbose": False,
    },
    kwargs_model={
        "beta": 0.5,
    },
    init_callback_fun=init_callback_fun,
    exp_name="chroGibbs_init_GUM_2",
    with_energy=False,
    with_jit=False,
    with_n_iter=True,
)

key, subkey = jax.random.split(key, 2)
Ks = jnp.arange(2, 3)
times, n_iterations, _ = time_complete_sampling(
    Sampler=ChromaticGibbsSampler,
    Model=Potts,
    key=subkey,
    Ks=Ks,
    sizes=sizes,
    reps=reps,
    kwargs_sampler={
        "eps": 0.05,
        "max_iter": 1000,
        "color_update_type": "sequential_in_color",
        "verbose": False,
    },
    kwargs_model={
        "beta": 0.5,
    },
    exp_name="chroGibbs_init_random_2",
    with_energy=False,
    with_jit=False,
    with_n_iter=True,
)

# K = 7


def init_callback_fun(key, K, lx, ly):
    gum = GUM(kappa=kappa, K=K, dim=2)
    gum_sampler = GUMSampler(n_bands=2500, lx=lx, ly=ly, method="spectral")
    return gum_sampler.sample_image(gum, subkey)[0]


key, subkey = jax.random.split(key, 2)
Ks = jnp.arange(7, 8)

times, n_iterations, _ = time_complete_sampling(
    Sampler=ChromaticGibbsSampler,
    Model=Potts,
    key=subkey,
    Ks=Ks,
    sizes=sizes,
    reps=reps,
    kwargs_sampler={
        "eps": 0.05,
        "max_iter": 1000,
        "color_update_type": "sequential_in_color",
        "verbose": False,
    },
    kwargs_model={
        "beta": 1.0,
    },
    init_callback_fun=init_callback_fun,
    exp_name="chroGibbs_init_GUM_7",
    with_energy=False,
    with_jit=False,
    with_n_iter=True,
)

key, subkey = jax.random.split(key, 2)
Ks = jnp.arange(7, 8)
times, n_iterations, _ = time_complete_sampling(
    Sampler=ChromaticGibbsSampler,
    Model=Potts,
    key=subkey,
    Ks=Ks,
    sizes=sizes,
    reps=reps,
    kwargs_sampler={
        "eps": 0.05,
        "max_iter": 1000,
        "color_update_type": "sequential_in_color",
        "verbose": False,
    },
    kwargs_model={
        "beta": 1.0,
    },
    exp_name="chroGibbs_init_random_7",
    with_energy=False,
    with_jit=False,
    with_n_iter=True,
)
