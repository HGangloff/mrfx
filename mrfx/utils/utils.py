import jax.numpy as jnp
from jaxtyping import Int, Array

def get_most_frequent_values(img_list: Int[Array, " x lx ly"], K: int) -> Int[Array, " lx ly"]:
    r"""
    img_list is a (n, lx, ly) list of images for example,
    we return the (lx, ly) array where we find the most
    frequent value at site i\in[1,lx],j\in[1,ly] among
    the n values in input

    K is the number of classes to expect
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