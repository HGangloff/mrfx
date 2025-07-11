"""
Abstract base class for iterative algorithms
"""

import abc
from typing import Any

from jaxtyping import Array
import jax
import jax.numpy as jnp
import equinox as eqx

from mrfx.utils.utils import get_most_frequent_values


class IterativeAlgorithm(eqx.Module):
    eps: float = eqx.field(kw_only=True)
    max_iter: int = eqx.field(kw_only=True)
    cv_type: str = eqx.field(static=True, default="avg_and_iter", kw_only=True)
    verbose: bool = eqx.field(default=True)

    @abc.abstractmethod
    def run(self, *_, **__) -> Any:
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

    def check_convergence_iter_only(self, _, __, iterations: int, ___) -> bool:
        """
        Function used to assess convergence in the run of a Gibbs sampler
        """

        return self.check_max_iter(iterations)

    def check_convergence_avg_and_iter(
        self, K: int, X_list: Array, iterations: int, _
    ) -> bool:
        """
        Function used to assess convergence in the run of a Gibbs sampler
        """

        def check_average(X_list):
            X_n_prev = jnp.array(X_list[:-1])
            X = X_list[-1]

            most_frequent_val = get_most_frequent_values(X_n_prev, K)
            return jax.lax.cond(
                (most_frequent_val != X).mean() < self.eps,
                lambda _: self.stop_while_loop_message(
                    "Convergence criterion is reached"
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
