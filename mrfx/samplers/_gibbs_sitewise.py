"""
Abstract method for Gibbs sampler variations
"""

import jax
import jax.numpy as jnp
from jax import jit
import equinox as eqx
from jaxtyping import Int, Key, Array

from mrfx.abstract._iterative_algorithm import IterativeAlgorithm
from mrfx.models._abstract_mrf import AbstractMarkovRandomFieldModel
from mrfx.abstract._sampler import AbstractSampler
from mrfx.samplers._utils import get_neigh


class GibbsSamplerSiteWise(AbstractSampler, IterativeAlgorithm):
    """
    Like GibbsSampler but at each step we do a single site update and not a
    whole image.
    This is useful for ESS computation

    CANNOT be used with check_cv_fun==check_convergence_avg_and_iter
    """

    name: str = eqx.field(static=True, kw_only=True, default="Gibbs sampler sitewise")

    def run(
        self,
        model: AbstractMarkovRandomFieldModel,
        key: Key,
        X_init: Array | None = None,
        keep_sample_list=False,
    ) -> tuple[Array, Array, Int]:
        # initialization
        key, subkey = jax.random.split(key, 2)

        samples = None

        if keep_sample_list:
            samples = []

        def insert_sample(sample):
            nonlocal samples
            samples.append(jax.device_put(sample, jax.devices("cpu")[0]))

        if X_init is None:
            X_init = jax.random.randint(
                subkey, (self.lx, self.ly), minval=0, maxval=model.K
            )

        # X_list = jax.random.randint(
        #    subkey, (self.n_it_for_cv + 1, self.lx, self.ly), minval=0, maxval=model.K
        # )
        # X_list = X_list.at[-1].set(X_init)

        if keep_sample_list:
            insert_sample(X_init)

        iterations = 0

        def body_fun(model, X, iterations, key):
            uv = iterations % (self.ly * self.lx)
            key, subkey = jax.random.split(key, 2)
            X_key, _ = self.update_one_site(X, subkey, model, uv)
            X = X_key[0]
            # X_list = jnp.roll(X_list, shift=-1, axis=0)
            # X_list = X_list.at[-1].set(X)
            iterations += 1

            if keep_sample_list:
                # here we need a non pure callback:
                # 1) not possible to know in advance when the loop will break,
                # thus it is too costly to preallocate max_iter images on GPU
                # and to use it in the carry
                # 2) so we need the sample list on the cpu but it is not
                # possible to have a carry which is both on GPU and CPU
                jax.experimental.io_callback(insert_sample, None, X)

            return (model, X, iterations, key)

        init_val = (model, X_init, iterations, key)
        model, X, iterations, key = jax.lax.while_loop(
            lambda args: self.check_cv_fun(*((args[0].K,) + args[1:])),
            lambda args: body_fun(*args),
            init_val,
        )

        if keep_sample_list:
            return X_init, X, iterations + 1, jnp.array(samples)

        return X_init, X, iterations + 1

    @jit
    def update_one_site(
        self,
        X: Array,
        key: Key,
        model: AbstractMarkovRandomFieldModel,
        uv: Int,
    ) -> tuple[tuple[Array, Key], None]:
        u, v = jnp.unravel_index(uv, (self.lx, self.ly))
        neigh_values = get_neigh(X, u, v, self.lx, self.ly, model.neigh_size)
        key, subkey = jax.random.split(key, 2)
        potential_values = model.potential_values(neigh_values, u, v)
        x_sample = model.sample(potential_values, key=subkey)
        X = X.at[u, v].set(x_sample)
        return (X, key), None

    @jit
    def update_one_image(self, *args, **kwargs):
        pass
