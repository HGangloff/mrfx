"""
Abstract method for Gibbs sampler variations
"""

import abc
import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import Int, Key, Array

from mrfx.abstract._iterative_algorithm import IterativeAlgorithm
from mrfx.models._abstract import AbstractMarkovRandomFieldModel
from mrfx.abstract._sampler import AbstractSampler


class AbstractGibbsSampler(AbstractSampler, IterativeAlgorithm):
    """ """

    def run(
        self, model: AbstractMarkovRandomFieldModel, key: Key
    ) -> tuple[Array, Array, Int]:
        # initialization
        key, subkey = jax.random.split(key, 2)
        X_init = jax.random.randint(
            subkey, (self.lx, self.ly), minval=0, maxval=model.K
        )

        key, key_permutation = jax.random.split(key, 2)

        n_it_for_cv = 10  # we use the pixelwise averaging over the last 10
        # iterations to assess convergence

        key, subkey = jax.random.split(key, 2)
        X_list = jax.random.randint(
            subkey, (n_it_for_cv + 1, self.lx, self.ly), minval=0, maxval=model.K
        )
        X_list = X_list.at[-1].set(X_init)
        iterations = 0

        if self.cv_type == "iter_only":
            check_cv_fun = self.check_convergence_iter_only
        elif self.cv_type == "avg_and_iter":
            check_cv_fun = self.check_convergence_avg_and_iter
        else:
            raise ValueError("Unrecognized value for cv_type")

        def body_fun(model, X_list, iterations, key):
            key, subkey = jax.random.split(key, 2)
            # key, key_permutation = jax.random.split(key, 2)
            X = self.update_one_image(X_list[-1], model, subkey, key_permutation)
            # jax.debug.print("{x}", x=X)
            X_list = jnp.roll(X_list, shift=-1, axis=0)
            X_list = X_list.at[-1].set(X)
            iterations += 1
            return (model, X_list, iterations, key)

        init_val = (model, X_list, iterations, key)
        model, X_list, iterations, key = jax.lax.while_loop(
            lambda args: check_cv_fun(*((args[0].K,) + args[1:])),
            lambda args: body_fun(*args),
            init_val,
        )

        return X_init, X_list, iterations + 1

    @jit
    @abc.abstractmethod
    def update_one_image(
        self,
        X: Array,
        model: AbstractMarkovRandomFieldModel,
        key: Key,
        key_permutation: Key,
    ) -> Array:
        """
        Receives X at previous Gibbs iteration
        Outputs an updated X

        X_full represents the full image, useful in the case where we have parallelization
        through coloring for example. Otherwise X_full = X
        """
        raise NotImplementedError
