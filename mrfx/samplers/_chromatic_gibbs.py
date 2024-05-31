"""
Chromatic Gibbs sampler
"""

import jax
import jax.numpy as jnp
from jaxtyping import Int, Float, Key, Array, Bool
import equinox as eqx
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

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
    devices: Array = eqx.field(init=False)
    mesh: Mesh = eqx.field(init=False)

    def __post_init__(self):
        if self.color_update_type in [
            "shard_map_and_sequential_in_color",
            "shard_map_and_vmap_in_color",
        ]:
            if self.n_devices is None:
                self.n_devices = jax.local_device_count()
            self.devices = mesh_utils.create_device_mesh((self.n_devices,))
            self.mesh = Mesh(self.devices, ("i",))
        else:
            self.devices = None
            self.mesh = None

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

        vmap_update_one_image = jax.vmap(
            self.update_one_color, (0, None, None, None, None, 0)
        )
        X_colors = vmap_update_one_image(
            X_colors, model, key, key_permutation, X, color_offset
        )
        X = X.at[0::2, 0::2].set(X_colors[0])
        X = X.at[1::2, 0::2].set(X_colors[1])
        X = X.at[0::2, 1::2].set(X_colors[2])
        X = X.at[1::2, 1::2].set(X_colors[3])
        return X

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
            def update_one_site_sequential(X, key, K, lx, ly, X_full, color_offset):
                """ """
                update_one_site_ = lambda carry, uv: update_one_site(
                    carry, uv, K, lx, ly, X_full, color_offset
                )
                carry, _ = jax.lax.scan(
                    update_one_site_,
                    (X, key),
                    jax.random.permutation(key_permutation, jnp.arange(n_sites)),
                )
                return carry[0]

            return update_one_site_sequential(X, key, K, lx, ly, X_full, color_offset)

        elif self.color_update_type == "shard_map_and_sequential_in_color":
            # Chromatic Gibbs sampler that performs a sequential update
            # on n=jax.local_device_count() parallel devices. A true parallel
            # beviour is obtained with shard_map
            def return_one_site_sequential(sites, key, K, lx, ly, X_full, color_offset):
                """
                Sequential return when we cannot parallelize anymore because
                all the device for parallelization have been taken
                """
                update_one_site_ = lambda carry, uv: return_one_site_scan(
                    carry, uv, K, lx, ly, X_full, color_offset
                )
                carry, X_flat = jax.lax.scan(update_one_site_, (key.squeeze(),), sites)
                return X_flat

            return_one_site_sequential_ = lambda sites, key: return_one_site_sequential(
                sites, key, K, lx, ly, X_full, color_offset
            )

            return_one_site_parallel = shard_map(
                return_one_site_sequential_,
                self.mesh,
                in_specs=(P("i"), P("i")),
                out_specs=P("i"),
            )
            X = return_one_site_parallel(
                site_permutation,
                jax.random.split(
                    key, self.n_devices
                ),  # one key for each parallel program
            )

        elif self.color_update_type == "shard_map_and_vmap_in_color":
            # Chromatic Gibbs sampler that performs a vmap update
            # on n=jax.local_device_count() parallel devices. A true parallel
            # beviour is obtained with shard_map
            return_one_site_ = lambda key, uv: return_one_site(
                key, uv, K, lx, ly, X_full, color_offset
            )

            # we use vmap when we cannot parallelize anymore (by lack of devices eg.)
            vmap_return_one_site_ = jax.vmap(return_one_site_, (0, 0))

            # paralellize with shard_map which can effectively be composed with vmap / JIT etc.
            return_one_site_parallel = shard_map(
                vmap_return_one_site_,
                self.mesh,
                in_specs=(P("i"), P("i")),
                out_specs=P("i"),
            )
            X = return_one_site_parallel(
                jax.random.split(
                    key, n_sites
                ),  # jax.random.split(key, jax.local_device_count()),
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
                jax.random.split(
                    key, n_sites
                ),  # jax.random.split(key, jax.local_device_count()),
                site_permutation,
            )

        else:
            raise ValueError("Wrong value for self.color_update_type")

        # Common for "shard_map_and_sequential_in_color",
        # "shard_map_and_vmap_in_color", "vmap_in_color":
        # invert the permutation to get X in good order for reshaping
        X = X[jnp.argsort(site_permutation)]
        return X.reshape(lx_color, ly_color)

    def return_one_site(
        self,
        key: Key,
        model: AbstractMarkovRandomFieldModel,
        uv: Int,
        lx_color: Int,
        ly_color: Int,
        X_full: Array,
        color_offset: tuple[Int, Int],
    ) -> Int:
        """ """
        u, v = jnp.unravel_index(uv, (lx_color, ly_color))
        u_full_scale, v_full_scale = (
            u * (X_full.shape[0] // lx_color) + color_offset[0],
            v * (X_full.shape[1] // ly_color) + color_offset[1],
        )
        neigh_values = get_neigh(
            X_full, u_full_scale, v_full_scale, X_full.shape[0], X_full.shape[1]
        )
        potential_values = model.potential_values(neigh_values)
        return model.sample(potential_values, key)
