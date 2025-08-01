mrfx
====

Old school discrete Markov Random Fields (and more) with JAX for regular lattices (images). Getting the most of JAX for the computational tasks

:warning: Under heavy development. Check out the notebooks

## Markov Random Fields

**Unsupervised segmentation**

<img src="./notebooks/real_example_segmentation.png" height="275" width="800" >


**Fast sampling**

<img src="./illustrations/MRF.png" height="250" width="250" >

<img src="./illustrations/time_update_one_image.png" height="250" width="350" >


## Discrete Gaussian Unitary Simplex

**Fast sampling**

<img src="./illustrations/DGUM.png" height="250" width="250" >


# Documentation

# Contributing

* First fork the library.

* Then clone and install the library in development mode with

```bash
pip install -e .
```

* Install pre-commit and run it.

```bash
pip install pre-commit
pre-commit install
```

* Open a merge request once you are done with your changes.

# Citing

```
@article{courbot2025gaussian,
  title={Gaussian Unit-simplex Markov random fields as a fast proxy for MRF sampling},
  author={Courbot, Jean-Baptiste and Gangloff, Hugo},
  year={2025}
}
```
