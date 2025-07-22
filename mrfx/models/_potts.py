"""
Potts model
"""

from dataclasses import InitVar
from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import equinox as eqx

from mrfx.models._abstract import AbstractMarkovRandomFieldModel
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

    def estimate_parameters(self, self_realization: Array):
        """
        Derin and Elliott 1987 least square estimator
        Modeling and segmentation of noisy and textured images using gibbs random fields 
        """
        lx, ly = self_realization.shape[0], self_realization.shape[1]
        config_xi_and_neighborhoods = jnp.unravel_index(
            jnp.arange(self.K ** (8 + 1)),
            (self.K,) * (8 + 1)
        )
        proba_xi_and_neighborhoods = jnp.zeros((self.K ** (8 + 1) + 1,))

        v_get_neigh = jax.vmap(get_neigh, (None, 0, None, None, None, None,
                                           None))
        vv_get_neigh = jax.vmap(v_get_neigh, (None, None, 0, None, None, None,
                                              None))
        xi_and_neigh_values_for_each_site = vv_get_neigh(
            self_realization, 
            jnp.arange(lx),
            jnp.arange(ly),
            lx,
            ly,
            self.neigh_size,
            True
        )
        def get_config_idx(config, config_type):
            if config_type == "xi_and_nei":
                idx = range(9)
                size = 2 ** 0
            if config_type == "nei":
                idx = range(1, 9)
                size = 2 ** 1
            if config_type == "xi":
                idx = range(0, 1)
                size = 2 ** 8
            arr = jnp.array(tuple(config_xi_and_neighborhoods[i] ==
                                  config[i] for i in idx))
            return jnp.where(
                jax.lax.reduce(arr, (True), jnp.logical_and,
                               (0,)), size=size
            )[0]
        v_get_config = jax.vmap(get_config_idx, (0, None))
        vv_get_config = jax.vmap(v_get_config, (0, None))

        ### 1) Estimate the empirical probabilities of xi+neighborhood
        # below we get, at each site, the id of the neigh config
        config_each_site = vv_get_config(
            xi_and_neigh_values_for_each_site, "xi_and_nei"
        )
        # NOTE the trick to handle results with varying sizes
        # -> we fill with self.K**8 which is an non existing neighborhood
        # config. We have prepared a room for 
        # proba_neighborhoods = jnp.zeros((self.K ** (8 + 1) + 1,))
        count_config = jnp.unique_counts(config_each_site, size=self.K**(8+1),
                                         fill_value=self.K**(8+1))
        proba_xi_and_neighborhoods = jnp.put(
            proba_xi_and_neighborhoods,
            count_config.values,
            count_config.counts,
            mode="clip",
            inplace=False
        ) / (lx * ly)

        ### 2) Get one equation for each (s,s', N) in Omega X Omega X Omega**8
        config_idx_xi_fixed = jnp.stack([
            get_config_idx(jnp.array([k]), "xi") for k in range(self.K)
        ])
        print(config_idx_xi_fixed.shape)
        fs

        # Summations for each site
        # Now, all neighborhoods are considered totally homogeneous wrt to all
        # directions hence, p(x_s, x_Ns) reduces to counting the number of
        # elements equal to x_s
        print(xi_and_neigh_values_for_each_site.shape)
        sums_delta = jnp.stack([
            jnp.sum(xi_and_neigh_values_for_each_site[..., 1:] == k, axis=-1) for k
            in range(self.K)], axis=0
        )


        
        print(proba_neighborhoods.shape, sums_delta.shape)
