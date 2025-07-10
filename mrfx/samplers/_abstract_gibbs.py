"""
Abstract method for Gibbs sampler variations
"""

import abc
import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import Int, Float, Key, Array, Bool
import equinox as eqx

from mrfx.models._abstract import AbstractMarkovRandomFieldModel
from mrfx.samplers._abstract import AbstractSampler
from mrfx.utils.utils import get_most_frequent_values


class AbstractGibbsSampler(AbstractSampler):
    """ """

    eps: Float = eqx.field(kw_only=True)
    max_iter: Int = eqx.field(kw_only=True)
    cv_type: str = eqx.field(static=True, default="avg_and_iter", kw_only=True)
    verbose: bool = eqx.field(default=True)

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
            lambda args: check_cv_fun(*args),
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

    def stop_while_loop_message(self, msg):
        jax.lax.cond(
            self.verbose,
            lambda _: jax.debug.print(f"Stopping Gibbs sampler, cause: {msg}"),
            lambda _: None,
            None,
        )
        return False

    def check_max_iter(self, i):
        return jax.lax.cond(
            i >= self.max_iter,
            lambda _: self.stop_while_loop_message("Max iterations reached"),
            lambda _: True,
            None,
        )

    def check_convergence_iter_only(self, _, __, iterations: Int, ___) -> Bool:
        """
        Function used to assess convergence in the run of a Gibbs sampler
        """

        return self.check_max_iter(iterations)

    def check_convergence_avg_and_iter(
        self, model: AbstractMarkovRandomFieldModel, X_list: Array, iterations: Int, _
    ) -> Bool:
        """
        Function used to assess convergence in the run of a Gibbs sampler
        """

        def check_average(X_list):
            X_n_prev = jnp.array(X_list[:-1])
            X = X_list[-1]

            most_frequent_val = get_most_frequent_values(X_n_prev, model.K)
            return jax.lax.cond(
                (most_frequent_val != X.flatten()).mean() < self.eps,
                lambda _: self.stop_while_loop_message(
                    "Convergence criterion " "is reached"
                ),  # Stop while loop when converged
                lambda _: True,  # Continue while loop when not converged
                (None,),
            )

        # maxval
        max_iter_cond = jnp.array(iterations >= self.max_iter)

        # the logical_or force stopping as soon as we reached max_iter even if
        # lower than n
        return jax.lax.cond(
            jnp.logical_or(jnp.array(iterations > X_list.shape[0] - 1), max_iter_cond),
            lambda args: jax.tree.reduce(
                lambda x, y: jnp.logical_and(jnp.array(x), jnp.array(y)),
                (check_average(args), self.check_max_iter(iterations)),
            ),
            lambda args: True,  # Force while to continue if we do not have at least 10 iterations
            X_list,
        )
