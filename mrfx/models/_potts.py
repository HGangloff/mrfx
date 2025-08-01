"""
Potts model
"""

from dataclasses import InitVar
from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import equinox as eqx

from mrfx.models._abstract_mrf import AbstractMarkovRandomFieldModel
from mrfx.abstract._distributions import AbstractPriorDistribution, Params
from mrfx.samplers._utils import get_neigh


class PottsParameter(Params):
    beta: Float


class Potts(AbstractPriorDistribution, AbstractMarkovRandomFieldModel):
    """ """

    params: PottsParameter = eqx.field(kw_only=True, default=None)
    K: int = eqx.field(static=True, kw_only=True)
    neigh_size: int = eqx.field(
        kw_only=True, default=1, static=True
    )  # default is 8 nearest neighbors
    beta: InitVar[float] = eqx.field(default=None)

    def __post_init__(self, beta=None):
        """
        We let the user specify the parameter either via params or beta, the
        latter being less verbose
        """
        if self.params is None and beta is None:
            raise ValueError("`params` or `beta` must be specified")
        if self.params is not None and beta is not None:
            raise ValueError("`params` or `beta` cannot be specified together")
        if self.params is None and beta is not None:
            self.params = PottsParameter(beta=beta)

    def potential(self, x: Array, neigh_values: Array, *args, **kwargs) -> Float:
        return self.params.beta * (x == neigh_values).sum(dtype=neigh_values.dtype)

    def estimate_parameters(self, self_realization: Array, method="farag"):
        """
        We implement two approaches

        1) "derin": We implement Derin and Elliott's least square estimator of the granularity
        coefficient of a Potts Random Field
        Derin and Elliott, 1987, Modeling and segmentation of noisy and textured
        images using Gibbs random fields, TPAMI
        Note that this implementation is very memory greedy. It does not scale
        when K > 2 if using double vmap

        2) "farag": We implement Farag estimator from Farag et al.,
        A Unified Framework for MAP Estimation in Remote Sensing Image
        Segmentation, 2005, TGRS.
        """
        assert self.neigh_size == 1  # methods are hardcoded for this one

        lx, ly = self_realization.shape[0], self_realization.shape[1]
        if method == "farag":
            Tn = lx * ly * 8

            def count_equal_neighbors(i, j):
                neigh_values = get_neigh(
                    self_realization, i, j, lx, ly, self.neigh_size
                )
                return (self_realization[i, j] == neigh_values).sum(
                    dtype=neigh_values.dtype
                )

            v_count_equal_neighbors = jax.vmap(count_equal_neighbors, (0, None))
            vv_count_equal_neighbors = jax.vmap(v_count_equal_neighbors, (None, 0))
            f_eq = (
                1
                / Tn
                * jnp.sum(vv_count_equal_neighbors(jnp.arange(lx), jnp.arange(ly)))
            )
            beta = self.K**2 / (self.K - 1) * (f_eq - 1 / self.K)

        elif method == "derin":
            config_xi_and_neighborhoods = jnp.unravel_index(
                jnp.arange(self.K ** (8 + 1)), (self.K,) * (8 + 1)
            )
            proba_xi_and_neighborhoods = jnp.zeros((self.K ** (8 + 1) + 1,))

            v_get_neigh = jax.vmap(get_neigh, (None, 0, None, None, None, None, None))
            vv_get_neigh = jax.vmap(
                v_get_neigh, (None, None, 0, None, None, None, None)
            )
            xi_and_neigh_values_for_each_site = vv_get_neigh(
                self_realization,
                jnp.arange(lx),
                jnp.arange(ly),
                lx,
                ly,
                self.neigh_size,
                True,
            )

            def get_config_idx(config, config_type):
                if config_type == "xi_and_nei":
                    idx = range(9)
                    size = 2**0
                if config_type == "nei":
                    idx = range(1, 9)
                    size = 2**1
                if config_type == "xi":
                    idx = range(0, 1)
                    size = 2**8
                arr = jnp.array(
                    tuple(config_xi_and_neighborhoods[i] == config[i] for i in idx)
                )
                # arr shape is (len(idx), 512), and we know that only
                # nb=size of them is (1, 1, 1, ..., 1)
                # eg there is only one config matching the 8 neighbors + xi
                # there are 2 configs matching the 8 neighbors etc
                return jnp.where(
                    jax.lax.reduce(arr, (True), jnp.logical_and, (0,)), size=size
                )[0]

            # NOTE the double vmapping below avoids looping but explodes the memory as
            # soon as K>2
            v_get_config = jax.vmap(get_config_idx, (0, None))

            # vv_get_config = jax.vmap(v_get_config, (0, None))
            # However a full loop approach takes a long time (despite it has no
            # memory issue)
            # Hence we chose an hybrid approach: looping on the rows and vmap on
            # columns
            def one_row_iteration(carry, i):
                (all_sites,) = carry
                config_on_row_i = v_get_config(all_sites[i], "xi_and_nei")
                return (all_sites,), config_on_row_i

            ### 1) Estimate the empirical probabilities of xi+neighborhood
            # below we get, at each site, the id of the neigh config
            # NOTE vv_get_config is too memory complex (see note above)
            # config_each_site = vv_get_config(
            #    xi_and_neigh_values_for_each_site, "xi_and_nei"
            # )
            # Hence loop version
            config_each_site = jax.lax.scan(
                one_row_iteration, (xi_and_neigh_values_for_each_site,), jnp.arange(lx)
            )[1].reshape((lx, ly))

            # NOTE the trick to handle results with varying sizes
            # -> we fill with self.K**8 which is an non existing neighborhood
            # config. We have prepared a room for
            # proba_neighborhoods = jnp.zeros((self.K ** (8 + 1) + 1,))
            count_config = jnp.unique_counts(
                config_each_site, size=self.K ** (8 + 1), fill_value=self.K ** (8 + 1)
            )
            proba_xi_and_neighborhoods = jnp.put(
                proba_xi_and_neighborhoods,
                count_config.values,
                count_config.counts,
                mode="clip",
                inplace=False,
            ) / (lx * ly)

            # Get one equation for each (s, N) in Omega X Omega**8
            # Below we get the self.K ** 8 combinations where xi==k, for all k
            config_idx_xi_fixed = jnp.stack(
                [get_config_idx(jnp.array([k]), "xi") for k in range(self.K)]
            )

            def get_a(config_idx_1, config_idx_2):
                return jnp.log(
                    proba_xi_and_neighborhoods[config_idx_1]
                    / proba_xi_and_neighborhoods[config_idx_2]
                )

            def get_b(config_idx_1, config_idx_2, k1, k2):
                config_1 = jnp.unravel_index(config_idx_1, (self.K,) * (8 + 1))
                config_2 = jnp.unravel_index(config_idx_2, (self.K,) * (8 + 1))
                return 2 * jnp.sum(jnp.stack(config_1[1:]) == k1) - 2 * jnp.sum(
                    jnp.stack(config_2[1:]) == k2
                )

            v_get_a = jax.vmap(get_a, (0, 0))
            v_get_b = jax.vmap(get_b, (0, 0, None, None))

            def get_a_and_b_for_pair_of_states(k1, k2):
                a = v_get_a(config_idx_xi_fixed[k1], config_idx_xi_fixed[k2])
                b = v_get_b(config_idx_xi_fixed[k1], config_idx_xi_fixed[k2], k1, k2)
                return a, b

            v_get_a_and_b = jax.vmap(get_a_and_b_for_pair_of_states, (0, None), (0, 0))
            vv_get_a_and_b = jax.vmap(v_get_a_and_b, (None, 0), (0, 0))

            a, b = vv_get_a_and_b(jnp.arange(self.K), jnp.arange(self.K))
            # we need to filter because we have nan where the proba is null because
            # of unseen value. Filter nan to 0 -> its like if those lines where
            # suppressed from the equations we gathered
            a_filtered = jnp.nan_to_num(a, nan=0, posinf=0, neginf=0).flatten()
            dot_a = jnp.dot(a_filtered, a_filtered)
            beta = 1 / dot_a * jnp.dot(a_filtered, b.flatten())
        else:
            raise ValueError("Wrong method")

        return PottsParameter(beta=beta)
