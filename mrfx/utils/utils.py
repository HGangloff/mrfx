from itertools import permutations
import jax
import jax.numpy as jnp
from jaxtyping import Int, Array


def find_permutation(pred: Array, truth: Array, K: int) -> tuple[Array, float]:
    """
    Find the label segmentation that results in the lowest error. We test all permutations
    """
    a = tuple(range(K))
    all_perms = jnp.array(tuple(permutations(a)))

    def get_err_pred_permuted(perm, pred):
        perm_pred = jnp.nansum(
            jnp.array(
                [jnp.where(pred == idx, label, 0) for idx, label in enumerate(perm)]
            ),
            axis=0,
        )
        return jnp.count_nonzero(perm_pred.flatten() != truth.flatten()) / truth.size

    v_get_err_pred_permuted = jax.vmap(get_err_pred_permuted, (0, None))
    errors = v_get_err_pred_permuted(all_perms, pred)

    lowest_err = jnp.min(errors)
    best_perm = all_perms[jnp.argmin(errors)]
    best_pred = jnp.nansum(
        jnp.array(
            [jnp.where(pred == idx, label, 0) for idx, label in enumerate(best_perm)]
        ),
        axis=0,
    )

    return best_pred, lowest_err


def get_most_frequent_values(
    img_list: Int[Array, " x lx ly"], K: int
) -> Int[Array, " lx ly"]:
    r"""
    img_list is a (n, lx, ly) list of images for example,
    we return the (lx, ly) array where we find the most
    frequent value at site i\in[1,lx],j\in[1,ly] among
    the n values in input

    K is the number of classes to expect
    We use a bincount applied along on axis
    We flatten the array because along_axis can only have one axis
    adapted for JAX from https://stackoverflow.com/a/12300214
    """
    u, indices = jnp.unique(
        img_list.reshape((img_list.shape[0], -1)).T, return_inverse=True, size=K
    )  # note the tranpose, we count on the axis of size n
    img_max = u[
        jnp.argmax(
            jnp.apply_along_axis(jnp.bincount, 1, indices, length=K),
            axis=1,
        )
    ].reshape(img_list.shape[1:])
    return img_max
