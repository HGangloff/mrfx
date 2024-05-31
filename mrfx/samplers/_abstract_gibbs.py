"""
Abstract method for Gibbs sampler variations
"""

import abc
import jax
import jax.numpy as jnp
from jaxtyping import Int, Float, Key, Array, Bool
import equinox as eqx
from mrfx.models._abstract import AbstractMarkovRandomFieldModel
from mrfx.samplers._utils import get_neigh


class AbstractGibbsSampler(eqx.Module):
    """ """

    lx: eqx.AbstractVar[Int] = eqx.field(static=True)
    ly: eqx.AbstractVar[Int] = eqx.field(static=True)
    eps: eqx.AbstractVar[Float]
    max_iter: eqx.AbstractVar[Int]

    def run(
        self, model: AbstractMarkovRandomFieldModel, key: Key
    ) -> tuple[Array, Array, Int]:
        # initialization
        key, subkey = jax.random.split(key, 2)
        X_init = jax.random.randint(
            subkey, (self.lx, self.ly), minval=0, maxval=model.K
        )

        key, key_permutation = jax.random.split(key, 2)

        iterations_stored = 20 + 1

        key, subkey = jax.random.split(key, 2)
        X_list = jax.random.randint(
            subkey, (iterations_stored, self.lx, self.ly), minval=0, maxval=model.K
        )
        X_list = X_list.at[-1].set(X_init)
        iterations = 0

        def body_fun(model, X_list, iterations, key):

            key, subkey = jax.random.split(key, 2)
            X = self.update_one_image(X_list[-1], model, subkey, key_permutation)
            X_list = jnp.roll(X_list, shift=-1, axis=0)
            X_list = X_list.at[-1].set(X)
            iterations += 1
            return (model, X_list, iterations, key)

        init_val = (model, X_list, iterations, key)
        model, X_list, iterations, key = jax.lax.while_loop(
            lambda args: self.check_convergence(*args),
            lambda args: body_fun(*args),
            init_val,
        )

        return X_init, X_list, iterations + 1

    @abc.abstractmethod
    def update_one_image(
        self,
        X: Array,
        model: AbstractMarkovRandomFieldModel,
        key: Key,
        key_permutation: Key,
        X_full: Array = None,
        color_offset: tuple[Int, Int] = None,
    ) -> Array:
        """
        Receives X at previous Gibbs iteration
        Outputs an updated X

        X_full represents the full image, useful in the case where we have parallelization
        through coloring for example. Otherwise X_full = X
        """
        raise NotImplementedError

    def check_convergence(
        self, model: AbstractMarkovRandomFieldModel, X_list: Array, iterations: Int, _
    ) -> Bool:
        """
        Function used to assess convergence in the run of a Gibbs sampler
        """

        def check_average(X_list):
            X_10_prev = jnp.array(X_list[:-1])
            X = X_list[-1]
            X_10_prev = X_10_prev.reshape(X_10_prev.shape[0], -1)
            # we find the most frequent value for each site over the 10 last simulations
            # we use a bincount applied along on axis
            # We flatten the array because along_axis can only have one axis
            u, indices = jnp.unique(
                X_10_prev.T, return_inverse=True, size=model.K
            )  # note the tranpose, we count on the axis of size 10
            most_frequent_val = u[
                jnp.argmax(
                    jnp.apply_along_axis(jnp.bincount, 1, indices, length=model.K),
                    axis=1,
                )
            ]  # adapted for JAX from https://stackoverflow.com/a/12300214
            return jax.lax.cond(
                (most_frequent_val != X.flatten()).mean() < self.eps,
                lambda _: False,  # Stop while loop when converged
                lambda _: True,  # Continue while loop when not converged
                (None,),
            )

        return jax.lax.cond(
            iterations > 10,
            lambda args: jnp.logical_and(
                check_average(args), jnp.array(iterations <= self.max_iter)
            ),
            lambda args: True,  # Force while to continue if we do not have at least 10 iterations
            X_list,
        )
