"""
Chromatic Gibbs sampler
"""

import jax
from jax import jit
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jaxtyping import Int, Float, Key, Array
import equinox as eqx

from mrfx.models._abstract import AbstractMarkovRandomFieldModel
from mrfx.samplers._abstract_gibbs import AbstractGibbsSampler
from mrfx.samplers._utils import get_neigh


class ChromaticGibbsSampler(AbstractGibbsSampler):
    """ """

    lx: Int = eqx.field(static=True)
    ly: Int = eqx.field(static=True)
    eps: Float
    max_iter: Int
    color_update_type: str = eqx.field(static=True)
    n_devices: Int = None

    # all the init=False fields are set in __post_init__
    devices: Array = eqx.field(static=True, init=False)
    mesh: Mesh = eqx.field(static=True, init=False)

    def __post_init__(self):
        if self.color_update_type in [
            "shard_map_then_sequential_in_color",
            "shard_map_then_vmap_in_color",
        ]:
            if self.n_devices is None:
                self.n_devices = jax.local_device_count()
            self.devices = mesh_utils.create_device_mesh((self.n_devices,))
            self.mesh = Mesh(self.devices, ("i",))
        else:
            self.devices = None
            self.mesh = None

    @jit
    def update_one_image(
        self,
        X: Array,
        model: AbstractMarkovRandomFieldModel,
        key: Key,
        key_permutation: Key,
    ) -> Array:
        """
        The update_one_image function of the ChromaticGibbsSampler is one level
        more abstract than the update_one_image of GibbsSampler since we pass
        through the step of dividing the image according to colors

        Then the real image update happens on the subimages of the colorings.
        This call is vectorized with vmap
        """
        X_colors = jnp.stack(
            [X[0::2, 0::2], X[1::2, 0::2], X[0::2, 1::2], X[1::2, 1::2]], axis=0
        )
        color_offset = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        keys = jax.random.split(key, 4)
        key_permutations = jax.random.split(key_permutation, 4)

        vmap_update_one_image = jax.vmap(
            self.update_one_color, (0, None, 0, 0, None, 0)
        )
        X_colors = vmap_update_one_image(
            X_colors, model, keys, key_permutations, X, color_offset
        )
        X = X.at[0::2, 0::2].set(X_colors[0])
        X = X.at[1::2, 0::2].set(X_colors[1])
        X = X.at[0::2, 1::2].set(X_colors[2])
        X = X.at[1::2, 1::2].set(X_colors[3])

        # X = self.update_one_color(
        #    X, model, key, key_permutation, X, (0,0)
        # )
        return X

    @jit
    def update_one_color(
        self,
        X: Array,
        model: AbstractMarkovRandomFieldModel,
        key: Key,
        key_permutation: Key,
        X_full: Array,
        color_offset: tuple[Int, Int],
    ) -> Array:
        """
        Receives X_full at previous Gibbs iteration
        Outputs an updated color X
        """
        lx_color, ly_color = X.shape[0], X.shape[1]
        n_sites = lx_color * ly_color  # note that this is lx, ly for the color!

        site_permutation = jax.random.permutation(key_permutation, jnp.arange(n_sites))

        if self.color_update_type == "sequential_in_color":
            # Chromatic Gibbs sampler that does not parallelize
            # inside a color and perform full sequential across a color
            X = self.return_sites_sequential(
                key,
                model,
                site_permutation,
                lx_color,
                ly_color,
                X_full,
                color_offset,
            )

        elif self.color_update_type == "shard_map_then_sequential_in_color":
            # Chromatic Gibbs sampler that performs a sequential update
            # on n=jax.local_device_count() parallel devices. A true parallel
            # beviour is obtained with shard_map
            # Sequential return when we cannot parallelize anymore because
            # all the device for parallelization have been taken

            # Note that we dont have lambda keys, sites: -> only one key per
            # sharding
            # because when passing a list of keys that we
            #  we do not converge to a good Gibbs sample
            # same is true for the sequential_in_color above (see comments in
            # self.return_sites_sequential...
            return_sites_sequential_ = lambda key, sites: self.return_sites_sequential(
                key.squeeze(), model, sites, lx_color, ly_color, X_full, color_offset
            )

            return_one_site_parallel = shard_map(
                return_sites_sequential_,
                self.mesh,
                in_specs=(P("i"), P("i")),
                out_specs=P("i"),
            )
            X = return_one_site_parallel(
                jax.random.split(key, jax.local_device_count()),
                site_permutation,
            )

        elif self.color_update_type == "shard_map_then_vmap_in_color":
            # Chromatic Gibbs sampler that performs a vmap update
            # on n=jax.local_device_count() parallel devices. A true parallel
            # beviour is obtained with shard_map
            # we use vmap when we cannot parallelize anymore (by lack of devices eg.)
            return_one_site_ = lambda key, uv: self.return_one_site(
                key, model, uv, lx_color, ly_color, X_full, color_offset
            )

            vmap_return_one_site_ = jax.vmap(return_one_site_, (0, 0))

            # paralellize with shard_map which can effectively be composed with vmap / JIT etc.
            return_one_site_parallel = shard_map(
                vmap_return_one_site_,
                self.mesh,
                in_specs=(P("i"), P("i")),
                out_specs=P("i"),
            )
            X = return_one_site_parallel(
                jax.random.split(key, n_sites),
                site_permutation,
            )

        elif self.color_update_type == "vmap_in_color":
            # Chromatic Gibbs sampler that performs vmap updates on the colors
            return_one_site_ = lambda key, uv: self.return_one_site(
                key, model, uv, lx_color, ly_color, X_full, color_offset
            )

            # we use vmap when we cannot parallelize anymore (by lack of devices eg.)
            vmap_return_one_site_ = jax.vmap(return_one_site_, (0, 0))

            X = vmap_return_one_site_(
                jax.random.split(key, n_sites),
                site_permutation,
            )

        else:
            raise ValueError("Wrong value for self.color_update_type")

        # Common for "shard_map_and_sequential_in_color",
        # "shard_map_and_vmap_in_color", "vmap_in_color":
        # invert the permutation to get X in good order for reshaping
        X = X[jnp.argsort(site_permutation)]
        return X.reshape(lx_color, ly_color)

    @jit
    def return_one_site(
        self,
        key: Key,
        model: AbstractMarkovRandomFieldModel,
        uv: Int,
        lx_color: Int,
        ly_color: Int,
        X_full: Array,
        color_offset: tuple[Int, Int],
    ) -> tuple[None, Int]:
        """ """
        u, v = jnp.unravel_index(uv, (lx_color, ly_color))
        u_full_scale, v_full_scale = (
            u * (self.lx // lx_color) + color_offset[0],
            v * (self.ly // ly_color) + color_offset[1],
        )
        neigh_values = get_neigh(X_full, u_full_scale, v_full_scale, self.lx,
                                 self.ly, model.neigh_size)
        potential_values = model.potential_values(neigh_values)
        return model.sample(potential_values, key)

    @jit
    def return_sites_sequential(
        self,
        key: Array,
        model: AbstractMarkovRandomFieldModel,
        sites: Array,
        lx_color: Int,
        ly_color: Int,
        X_full: Array,
        color_offset: tuple[Int, Int],
    ) -> Array:
        """

        Note that we do not have correct convergence if we pass a series of
        keys that we iterate over with scan without using the carry
        (see the code at the end of the function)
        """

        def return_one_site_wrapper(key, site):
            """
            Transform return_one_site so that it can be used in a scan function
            """
            key, subkey = jax.random.split(key)
            return key, self.return_one_site(
                subkey,
                model,
                site,
                lx_color,
                ly_color,
                X_full,
                color_offset,
            )

        _, X = jax.lax.scan(return_one_site_wrapper, key, sites)

        # code when `key` is an array of keys that we would like to iterate
        # over (the cause is not understood, in practice, Gibbs sampler do not
        # converge well...)
        # return_one_site_ = lambda carry, key_uv: (None, self.return_one_site(
        #    # carry, model, key_uv, lx_color, ly_color, X_full, color_offset
        #    key_uv[:2],
        #    model,
        #    key_uv[2],
        #    lx_color,
        #    ly_color,
        #    X_full,
        #    color_offset,
        # ))
        # _, X = jax.lax.scan(
        #    return_one_site_,
        #    None,
        #    jnp.concatenate([keys, sites[..., None]], axis=1).astype(
        #        jnp.uint32
        #    ),  # note the conversion that has no
        #    # effect for sites but which is needed to preserve the good
        #    # type for keys
        # )
        return X
