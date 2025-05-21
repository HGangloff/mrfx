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


class ExactSampler(AbstractSampler):
    """
    Reeves and Pettitt (2004) and Friel and Rue (2007) recursive strategy for
    exact sampling on relatively small lattices
    """

    def run(
        self, model: AbstractMarkovRandomFieldModel, key: Key
    ) -> tuple[Array, Array, Int]:

        # compute z recursion

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

    def z_recursion(
        self,
        model: AbstractMarkovRandomFieldModel,
    ) -> tuple[
        Float[Array, " (lx*ly-lag)*K**(neigh_size/2)"],  # noqa: F722
        Float[Array, ""],  # noqa: F722
    ]:
        dtype = jnp.float32

        if model.neigh_size == 1:
            n_neighbors = 8
            lag = self.lx + 1
        else:
            raise ValueError("Problem with model.neigh_size")

        # Homogeneous case
        nei_configurations = jnp.meshgrid(
            *((jnp.arange(model.K),) * (n_neighbors // 2 + 1))
        )
        nei_configurations = jnp.stack(
            jax.tree.map(jnp.ravel, nei_configurations), axis=-1
        )

        # and we will find the line correspong to a specific
        # configuration (for a specific z[i]):
        # jnp.ravel_multi_index(jax.random.randint(key, (K,), 0, 2), (K,)*lag)
        # v_ravel_multi_index = jax.vmap(jnp.ravel_multi_index, (0, None))

        # or we can find the configuration for a specific line with
        # unravel_index
        lag_config = jnp.unravel_index(jnp.arange(model.K**lag), (model.K,) * lag)

        def get_lag_configurations_idx_for_a_nei_configuration(
            nei_configuration,
            config_type,
        ):
            # 0 refers to the site itself
            # 1 refers to the bottom nei
            # 2 refers to the top right nei
            # 3 refers to right nei
            # 4 refers to bot right nei
            if config_type == "xi_and_nei":
                idx1 = [0, -4, -3, -2, -1]
                idx2 = [0, 1, 2, 3, 4]
                size = model.K ** (lag - 5)
            elif config_type == "xi_and_nei_no_last":
                idx1 = [0, -4, -3, -2]
                idx2 = [0, 1, 2, 3]
                size = model.K ** (lag - 4)
            elif config_type == "nei_only":
                idx1 = [-4, -3, -2, -1]
                idx2 = [1, 2, 3, 4]
                size = model.K ** (lag - 4)
            elif config_type == "xi_only":
                idx1 = idx2 = [0]
                size = model.K ** (lag - 1)
            else:
                raise ValueError("Wrong value for config_type")

            arr = jnp.array(
                tuple(
                    lag_config[i] == nei_configuration[ii] for i, ii in zip(idx1, idx2)
                )
            )
            lag_config_idx = jnp.where(
                jax.lax.reduce(arr, (True), jnp.logical_and, (0,)), size=size
            )[0]

            return lag_config_idx

        v_get_lag_configurations = jax.vmap(
            get_lag_configurations_idx_for_a_nei_configuration, (0, None)
        )
        lag_configurations_idx_with_xi = v_get_lag_configurations(
            nei_configurations, "xi_and_nei"
        )
        assert lag_configurations_idx_with_xi.shape[0] == 2**5
        assert 2**5 * lag_configurations_idx_with_xi.shape[1] == model.K**lag

        lag_configurations_idx_xi_no_last_nei = v_get_lag_configurations(
            nei_configurations, "xi_and_nei_no_last"
        )
        assert lag_configurations_idx_xi_no_last_nei.shape[0] == 2**5
        assert 2**4 * lag_configurations_idx_xi_no_last_nei.shape[1] == model.K**lag

        lag_configurations_idx_no_xi = v_get_lag_configurations(
            nei_configurations, "nei_only"
        )
        assert lag_configurations_idx_no_xi.shape[0] == 2**5
        assert 2**4 * lag_configurations_idx_no_xi.shape[1] == model.K**lag

        # lag_configurations_idx_xi_only = v_get_lag_configurations(
        #    nei_configurations,
        #    "xi_only"
        # )
        # assert lag_configurations_idx_xi_only.shape[0] == 2 ** 4
        # assert 2 * lag_configurations_idx_xi_only.shape[1] == model.K ** lag

        lag_configurations_idx_xi_0 = v_get_lag_configurations(
            jnp.array([[0]]), "xi_only"
        ).squeeze()
        lag_configurations_idx_xi_1 = v_get_lag_configurations(
            jnp.array([[1]]), "xi_only"
        ).squeeze()
        assert len(lag_configurations_idx_xi_0) == model.K ** (lag - 1)
        assert len(lag_configurations_idx_xi_1) == model.K ** (lag - 1)

        # init z: z[0]
        # lag_configurations_idx above represents where the 16 values
        # will be dispatched to the whole z[0]
        # now we look for the 16 values
        v_q = jax.vmap(
            lambda *args: model.potential(*args), (0, 0)
        )  # vmap on configurations
        v_v_q = jax.vmap(v_q, (0, None))  # vmap on states of xi
        xis = (
            jnp.ones(
                (
                    1,
                    nei_configurations.shape[0],
                ),
                dtype=dtype,
            )
            * jnp.arange(model.K, dtype=dtype)[:, None]
        )
        # NOTE xi is in nei_configurations but DO NOT count in the potential
        # hence the -1 for the column
        nei_configurations = nei_configurations.at[:, 0].set(-1)
        q = v_v_q(xis, nei_configurations.astype(dtype))

        # q in other configurations (top row, bottom row and right column)
        nei_configurations_top = nei_configurations.at[:, 2].set(-1)
        q_top = v_v_q(xis, nei_configurations_top.astype(dtype))
        # nei_configurations_right = nei_configurations.at[:, 2].set(-1)
        # nei_configurations_right = nei_configurations_right.at[:, 3].set(-1)
        # nei_configurations_right = nei_configurations_right.at[:, 4].set(-1)
        # q_right = v_v_q(xis, nei_configurations_right.astype(dtype))
        nei_configurations_bot = nei_configurations.at[:, 1].set(-1)
        nei_configurations_bot = nei_configurations_bot.at[:, 4].set(-1)
        q_bot = v_v_q(xis, nei_configurations_bot.astype(dtype))

        z0_values = jax.scipy.special.logsumexp(
            q_top, axis=0
        )  # those are the 32 values
        # z0_values = jnp.sum(q, axis=0)  # those are the 32 values

        # dispatch the values
        z0 = jnp.zeros((model.K**lag,), dtype=dtype)
        for i in range(lag_configurations_idx_with_xi.shape[0]):
            z0 = z0.at[lag_configurations_idx_with_xi[i]].set(z0_values[i])
        assert jnp.sum(z0 > 0) == model.K**lag  # assert we have filled all

        def scan_fun(carry, i):
            (z_i_1,) = carry

            # We need to choose q_at_i
            q_at_i = jax.lax.cond(
                (i % self.lx != 0) & (i % self.lx != (self.lx - 1)),
                lambda _: q,
                lambda _: jax.lax.cond(
                    i % self.lx == 0, lambda _: q_top, lambda _: q_bot, None
                ),
                None,
            )

            # vmap version
            def update_for_a_nei_config(z_i_1, lag_config_idx_xi_no_last_nei, q_at_i):
                z_idx0 = jnp.intersect1d(
                    lag_configurations_idx_xi_0,
                    lag_config_idx_xi_no_last_nei,
                    assume_unique=True,
                    size=model.K ** (lag - 5),
                )  # divide the size of lag_configurations_idx_no_xi_no_last_nei
                # by K since we intersect
                z_idx1 = jnp.intersect1d(
                    lag_configurations_idx_xi_1,
                    lag_config_idx_xi_no_last_nei,
                    assume_unique=True,
                    size=model.K ** (lag - 5),
                )

                z_ = jnp.stack([z_i_1[z_idx0], z_i_1[z_idx1]], dtype=z_i_1.dtype)
                # at all the following indices of z_i_1, ...

                return jax.scipy.special.logsumexp(q_at_i[:, None] + z_, axis=0)

            updates = jax.vmap(update_for_a_nei_config, (None, 0, 1))(
                z_i_1, lag_configurations_idx_xi_no_last_nei, q_at_i
            )
            for i in range(lag_configurations_idx_with_xi.shape[0]):
                z_i_1 = z_i_1.at[lag_configurations_idx_with_xi[i]].set(updates[i])

            z_i = z_i_1

            ## for version
            # z_i = z_i_1.copy()
            # for i in range(lag_configurations_idx_with_xi.shape[0]):
            #    # need those intersections to explicitly handle the state of xi
            #    z_idx0 = jnp.intersect1d(
            #        lag_configurations_idx_xi_0,
            #        lag_configurations_idx_xi_no_last_nei[i],
            #        assume_unique=True,
            #        size=model.K ** (lag - 4),
            #    )  # divide the size of lag_configurations_idx_no_xi_no_last_nei
            #    # by K since we intersect
            #    z_idx1 = jnp.intersect1d(
            #        lag_configurations_idx_xi_1,
            #        lag_configurations_idx_xi_no_last_nei[i],
            #        assume_unique=True,
            #        size=model.K ** (lag - 4),
            #    )
            #    z_ = jnp.stack([z_i_1[z_idx0], z_i_1[z_idx1]],
            #                   dtype=z_i_1.dtype)
            #    # at all the following indices of z_i_1, ...
            #    z_i = z_i.at[lag_configurations_idx_with_xi[i]].set(
            #        jax.scipy.special.logsumexp(q[:, i: i + 1] + z_, axis=0)
            #    )
            #    # z_i_1 = z_i_1.at[lag_configurations_idx_no_xi[i]].set(
            #    #    jnp.sum(q_at_i[:, i : i + 1] * z_, axis=0)
            #    # )

            return (z_i,), z_i

        _, all_z_i = jax.lax.scan(
            scan_fun, (z0,), jnp.arange(1, self.lx * self.ly - lag)
        )
        z = jax.scipy.special.logsumexp(all_z_i[-1])
        print(all_z_i[-1, :200])
        print(z)
        print(all_z_i.dtype)
        print(nei_configurations.shape)
        print(z0, z0.shape)

        return z, all_z_i

    @jit
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
