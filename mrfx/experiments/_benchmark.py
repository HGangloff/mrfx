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

from mrfx.models._abstract_mrf import AbstractMarkovRandomFieldModel
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
    exp_name=None,
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
    exp_name
        A string which names the experiment and the file where the results of
        the experiment will be stored

    Note that Ks must be a numpy array (because it is passed as a static_argnum
    in the jitted functions of mrfx and jax.numpy arrays are non hashable)
    """
    Ks = np.asarray(Ks)
    times = []
    for k in Ks:
        times.append([])
        model = Model(K=k, **kwargs_model)
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
                print(f"{r + 1} ", end="")
            runtime_mean = np.mean(rep_times)
            print(f"\n{k=}, {lx=}, {ly=}, {compilation_time=}, {runtime_mean=}")

            times[-1].append(runtime_mean)
    if exp_name is not None:
        df = pd.DataFrame(
            {"size": [lx * ly for lx, ly in sizes]}
            | {Ks[i]: times[i] for i in range(len(Ks))}
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
    with_n_iter=True,
    with_energy=False,
    with_jit=False,
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
    exp_name
        A string which names the experiment and the file where the results of
        the experiment will be stored
    return_X
        A bool to determine whether the sampled images are returned for further
        computations
    with_n_iter
        A bool to determine whether we keep track of the n_iter argument that
        some IterativeAlgorithm can output. False for GUMs of course.
    with_energy
        A bool to determine whether we also make statistics for the model
        energy consumption in Joules. **Requires `zeus` to be installed and JAX
        to run on GPU with idx `0`.**
    with_jit
        A bool to determine whether the run function is jitted here. Needs to
        be done for GUMs e.g.

    Note that Ks must be a numpy array (because it is passed as a static_argnum
    in the jitted functions of mrfx and jax.numpy arrays are non hashable)
    """
    if with_energy:
        from zeus.monitor import ZeusMonitor

        # NOTE that we assume that that the GPU is the device and that the GPU idx is 0
        monitor = ZeusMonitor(gpu_indices=[0], sync_execution_with="jax")

    if with_n_iter:
        n_iterations = []
    else:
        n_iterations = None
    Ks = np.asarray(Ks)
    times = []
    times_std = []
    if with_energy:
        energies = []
        energies_std = []
    else:
        energies = None
        energies_std = None
    if return_X:
        samples = []
    else:
        samples = None
    for k in Ks:
        times.append([])
        times_std.append([])
        if with_n_iter:
            n_iterations.append([])
        if with_energy:
            energies.append([])
            energies_std.append([])
        model = Model(K=k, **kwargs_model)
        if return_X:
            samples.append([])
        for lx, ly in sizes:
            sampler = Sampler(lx=lx, ly=ly, **kwargs_sampler)

            rep_times = []
            rep_iterations = []
            rep_energy = []
            print(f"Rep ( / {reps}): ", end="")
            key, subkey = jax.random.split(key, 2)

            start = time.time()
            if with_jit:
                run_fun = jax.jit(sampler.run)
            else:
                run_fun = sampler.run
            X_init, _, _ = run_fun(model, subkey)
            X_init.block_until_ready()
            end = time.time()
            compilation_time = end - start

            if return_X:
                samples[-1].append([])
            for r in range(reps):
                key, subkey = jax.random.split(key, 2)

                start = time.time()
                if with_energy:
                    monitor.begin_window("sampling")
                X_init, X_list, n_iter = run_fun(model, subkey)
                if with_energy:
                    measurement = monitor.end_window("sampling")
                X_init.block_until_ready()
                end = time.time()
                runtime = end - start

                rep_times.append(runtime)
                if return_X:
                    samples[-1][-1].append(X_list[-1])
                if with_n_iter:
                    rep_iterations.append(n_iter)
                if with_energy:
                    energy = next(iter(measurement.gpu_energy.values()))
                    rep_energy.append(energy)
                print(f"{r + 1} ", end="")

            times[-1].append(np.mean(rep_times))
            times_std[-1].append(np.std(rep_times))
            print(
                f"\n{k=}, {lx=}, {ly=}, {compilation_time=},"
                f"runtime_mean={times[-1][-1]}, ",
                end="",
            )

            if with_n_iter:
                n_iterations[-1].append(np.mean(rep_iterations))
                print(
                    f"n_iter_mean={n_iterations[-1][-1]}, ",
                    end="",
                )

            if with_energy:
                energies[-1].append(np.mean(rep_energy))
                energies_std[-1].append(np.std(rep_energy))

                print(f"energy_mean={energies[-1][-1]}")

    if exp_name is not None:
        df = pd.DataFrame(
            {"size": [lx * ly for lx, ly in sizes]}
            | {Ks[i]: times[i] for i in range(len(Ks))}
        )
        df.to_csv(f"{exp_name}_time.csv", index=False)
        df = pd.DataFrame(
            {"size": [lx * ly for lx, ly in sizes]}
            | {Ks[i]: times_std[i] for i in range(len(Ks))}
        )
        df.to_csv(f"{exp_name}_time_std.csv", index=False)
        if with_energy:
            df = pd.DataFrame(
                {"size": [lx * ly for lx, ly in sizes]}
                | {Ks[i]: energies[i] for i in range(len(Ks))}
            )
            df.to_csv(f"{exp_name}_energy.csv", index=False)
            df = pd.DataFrame(
                {"size": [lx * ly for lx, ly in sizes]}
                | {Ks[i]: energies_std[i] for i in range(len(Ks))}
            )
            df.to_csv(f"{exp_name}_energy_std.csv", index=False)
        return times, n_iterations, samples, energies
    return times, n_iterations, samples


def plot_benchmark(
    Ks: list,
    sizes: list,
    series: list,
    title: str,
    ylabel: str,
    fontsize: int = 10,
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
    series
        A list of list for the data to plot at each pair of k and size. Global list
        must be of len(Ks) and inner lists must of len(sizes)
    title
        Optional string for the title of the plot
    ylabel
        String for y axis
    """
    plt.rcParams.update({"font.size": fontsize})
    fig, axes = plt.subplots(1, 1)
    axes = [axes]
    for idx_k, times_for_k in enumerate(series):
        x = jnp.array([prod(s) for s in sizes])
        axes[0].plot(x, times_for_k, label=Ks[idx_k])
    axes[0].set_xlabel("number of sites")
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    if title is not None:
        plt.title(title)
    plt.savefig(f"{title}.pdf")
    plt.show()
