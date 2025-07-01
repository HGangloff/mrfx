"""
Code to benchmark the samplers
"""

import time
from math import prod
from typing import Type
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jaxtyping import Int, Key, Array
from numpy.typing import ArrayLike

from mrfx.models._abstract import AbstractMarkovRandomFieldModel
from mrfx.samplers._abstract_gibbs import AbstractGibbsSampler


def time_update_one_image(
    Sampler: Type[AbstractGibbsSampler],
    Model: Type[AbstractMarkovRandomFieldModel],
    key: Key,
    Ks: ArrayLike,
    sizes: Array,
    reps: Int,
    kwargs_sampler=None,
    kwargs_model=None,
    exp_name=None
) -> list:
    """
    Get a time estimate of the call to `update_one_image` for a given sampler
    and a given model

    Parameters
    ----------
    Sampler
        A non-instantiated AbstractGibbsSampler
    Model
        A non-instantiated AbstractMarkovRandomFieldModel
    key
        A jax PRNG key
    Ks
        A list, sequence, numpy array of the numbers of classes of the model
        that we want to time
    size
        An list, sequence, array of pairs of each (lx, ly), ie. the dimensions
        over which we want to time the sampler
    reps
        An integer. The number of time sample we average for a given
        combination of K and size
    kwargs_sampler
        A dictionary with the remaining arguments needed to instanciate the
        sampler (passed as keywords arguments): all the arguments apart from
        `lx` and `ly`
    kwargs_model
        A dictionary with the remaining arguments needed to instanciate the
        model (passed as keywords arguments): all the arguments apart from
        `K`

    Note that Ks must be a numpy array (because it is passed as a static_argnum
    in the jitted functions of mrfx and jax.numpy arrays are non hashable)
    """
    Ks = np.asarray(Ks)
    times = []
    for k in Ks:
        times.append([])
        model = Model(k, **kwargs_model)
        for lx, ly in sizes:
            sampler = Sampler(lx=lx, ly=ly, **kwargs_sampler)

            rep_times = []
            print(f"Rep ( / {reps}), ", end="")
            key, subkey = jax.random.split(key, 2)
            X = jax.random.randint(subkey, (lx, ly), minval=0, maxval=k)
            key, key_permutation = jax.random.split(key, 2)

            # compilation
            start = time.time()
            X = sampler.update_one_image(X, model, key, key_permutation)
            X.block_until_ready()
            end = time.time()
            compilation_time = end - start

            for r in range(reps):
                key, subkey = jax.random.split(key, 2)
                X = jax.random.randint(subkey, (lx, ly), minval=0, maxval=k)
                key, key_permutation = jax.random.split(key, 2)

                start = time.time()
                X = sampler.update_one_image(X, model, key, key_permutation)
                X.block_until_ready()
                end = time.time()
                runtime = end - start

                rep_times.append(runtime)
                print(f"{r+1} ", end="")
            runtime_mean = np.mean(rep_times)
            print(f"\n{k=}, {lx=}, {ly=}, {compilation_time=}, {runtime_mean=}")

            times[-1].append(runtime_mean)
    if exp_name is not None:
        df = pd.DataFrame(
            {"size":[lx*ly for lx, ly in sizes]} |
            {Ks[i]:times[i] for i in range(len(Ks))}
        )
        df.to_csv(f"{exp_name}.csv", index=False)
    return times


def time_complete_sampling(
    Sampler: Type[AbstractGibbsSampler],
    Model: Type[AbstractMarkovRandomFieldModel],
    key: Key,
    Ks: ArrayLike,
    sizes: Array,
    reps: Int,
    kwargs_sampler=None,
    kwargs_model=None,
    exp_name=None,
    return_X=False,
) -> tuple[list, list]:
    """
    Get a time estimate of the call to `update_one_image` for a given sampler
    and a given model

    Parameters
    ----------
    Sampler
        A non-instantiated AbstractGibbsSampler
    Model
        A non-instantiated AbstractMarkovRandomFieldModel
    key
        A jax PRNG key
    Ks
        A list, sequence, numpy array of the numbers of classes of the model
        that we want to time
    sizes
        A list, sequence, array of pairs of each (lx, ly), ie. the dimensions
        over which we want to time the sampler
    reps
        An integer. The number of time sample we average for a given
        combination of K and size
    kwargs_sampler
        A dictionary with the remaining arguments needed to instanciate the
        sampler (passed as keywords arguments): all the arguments apart from
        `lx` and `ly`
    kwargs_model
        A dictionary with the remaining arguments needed to instanciate the
        model (passed as keywords arguments): all the arguments apart from
        `K`

    Note that Ks must be a numpy array (because it is passed as a static_argnum
    in the jitted functions of mrfx and jax.numpy arrays are non hashable)
    """
    n_iterations = []
    Ks = np.asarray(Ks)
    times = []
    if return_X:
        samples = []
    else:
        samples = None
    for k in Ks:
        times.append([])
        n_iterations.append([])
        model = Model(k, **kwargs_model)
        if return_X:
            samples.append([])
        for lx, ly in sizes:
            sampler = Sampler(lx=lx, ly=ly, **kwargs_sampler)

            rep_times = []
            rep_iterations = []
            print(f"Rep ( / {reps}): ", end="")
            key, subkey = jax.random.split(key, 2)

            start = time.time()
            X_init, _, _ = sampler.run(model, subkey)
            X_init.block_until_ready()
            end = time.time()
            compilation_time = end - start

            if return_X:
                samples[-1].append([])
            for r in range(reps):
                key, subkey = jax.random.split(key, 2)

                start = time.time()
                X_init, X_list, n_iter = sampler.run(model, subkey)
                X_init.block_until_ready()
                end = time.time()
                runtime = end - start

                rep_times.append(runtime)
                if return_X:
                    samples[-1][-1].append(X_list[-1])
                rep_iterations.append(n_iter)
                print(f"{r+1} ", end="")
            runtime_mean = np.mean(rep_times)
            n_iter_mean = np.mean(rep_iterations)
            print(
                f"\n{k=}, {lx=}, {ly=}, {compilation_time=}, {runtime_mean=}, {n_iter_mean=}"
            )

            times[-1].append(runtime_mean)
            n_iterations[-1].append(n_iter_mean)
    if exp_name is not None:
        df = pd.DataFrame(
                {"size":[lx*ly for lx, ly in sizes]} |
                {Ks[i]:times[i] for i in range(len(Ks))}
            )
        df.to_csv(f"{exp_name}.csv", index=False)
    return times, n_iterations, samples


def plot_benchmark(
    Ks: list, sizes: list, times: list, n_iterations: list = None, title: str =
    None, fontsize: int = 10
):
    """

    Parameters
    ----------
    Ks
        A list, sequence, numpy array of the numbers of classes of the model
        that we want to time
    sizes
        A list, sequence, array of pairs of each (lx, ly), ie. the dimensions
        over which we want to time the sampler
    times
        A list of list for the timings at each pair of k and size. Global list
        must be of len(Ks) and inner lists must of len(sizes)
    n_iterations
        A list of list for the number of iterations at each pair of k and size.
        Global list must be of len(Ks) and inner lists must of len(sizes). Can
        be None.
    title
        Optional string for the title of the plot
    """
    plt.rcParams.update({'font.size': fontsize})
    if n_iterations is not None:
        fig, axes = plt.subplots(1, 2)
    else:
        fig, axes = plt.subplots(1, 1)
        axes = [axes]
    for idx_k, times_for_k in enumerate(times):
        x = jnp.array([prod(s) for s in sizes])
        axes[0].plot(x, times_for_k, label=Ks[idx_k])
    if n_iterations is not None:
        for idx_k, iterations_for_k in enumerate(n_iterations):
            x = jnp.array([prod(s) for s in sizes])
            axes[1].plot(x, iterations_for_k, label=Ks[idx_k])
            axes[1].set_xlabel("number of sites")
            axes[1].set_ylabel("number of iterations")
            axes[1].legend()
    axes[0].set_xlabel("number of sites")
    axes[0].set_ylabel("time (s)")
    axes[0].legend()
    if title is not None:
        plt.title(title)
    plt.savefig(f"{title}.pdf")
    plt.show()
