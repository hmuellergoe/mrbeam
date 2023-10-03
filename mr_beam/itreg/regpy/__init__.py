r"""# Python tools for regularization methods

A library for implementing and solving ill-posed inverse problems developed at the [Institute for
Numerical and Applied Mathematics Goettingen](https://num.math.uni-goettingen.de).

!!! Warning
    This project is alpha quality software and under heavy development. Excpect bugs and sparse
    documentation.

## Usage examples

To get an impression of how using `regpy` looks, there are some examples it the [`examples`
folder on GitHub](https://github.com/regpy/regpy/tree/release/examples), as well as inside the
release tarballs (see below).

## Test it with [Singularity](https://sylabs.io/singularity/)

If you're on Linux, you can use a premade Ubuntu-based Singularity container that contains
`regpy` and all of its dependencies and examples. Install instructions for Singularity can be
found [here][1].

[1]: https://sylabs.io/guides/3.4/user-guide/installation.html#distribution-packages-of-singularity

To get the container image, use

~~~ bash
singularity pull library://cruegge/default/regpy:latest
~~~

To run a shell inside the container, use

~~~ bash
singularity shell regpy_latest.sif
~~~

Note that the container's Python command is `python3`, not `python`. The examples can be found
under `/opt/examples`.

## Installation

### Obtaining the source code

- The source code is on GitHub ([regpy/regpy](https://github.com/regpy/regpy)).
- Releases are at the corresponding [release page](https://github.com/regpy/regpy/releases). The
  current version is 0.1.

### Dependencies

- `numpy >= 1.14`
- `scipy >= 1.1`

#### Optional dependencies

- [`ngsolve`](https://ngsolve.org/) (for some forward operators that require solving PDEs)
- [`bart`](https://mrirecon.github.io/bart/) (for the MRI operator)
- `pynfft >= 1.3` (for the iterative born solver for medium scattering)
- `matplotlib` (for some of the examples)
- [`pdoc3`](https://pdoc3.github.io/pdoc) (for generating the documentation)

### Installation with `pip`

A basic `setup.py` is provided, but the package is not on PyPI yet. To install it, clone this
repository or download the release tarball, and run

~~~ bash
pip install .
~~~

from the project's root folder. If you want to modify `regpy` itself, you can use

~~~ bash
pip install --editable .
~~~

to have Python load `regpy` from your current directory rathan than copying it to its library
folder.

In both cases, you can add `--no-deps` to prevent installing dependencies automatically,
in case you want to install them via some other package manager.
"""

from regpy import discrs, functionals, hilbert, mcmc, operators, solvers, stoprules, util

hilbert._register_spaces()
functionals._register_functionals()
