# `BFit`

[![This project supports Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org/downloads)
[![GitHub Actions CI Tox Status](https://github.com/theochem/bfit/actions/workflows/ci_tox.yml/badge.svg?branch=master)](https://github.com/theochem/bfit/actions/workflows/ci_tox.yml)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/theochem/bfit/master?labpath=%2Fexamples%2F)

## About

BFit is a Python library for fitting a convex sum of Gaussian functions to any
probability distribution. It is primarily intended for quantum chemistry applications, where the
basis functions are Gaussians and the fitted probability distribution is a scalar function like
the electron density.

See the example [section](#kl-fpi-models-of-atomic-densities) down below or the interactive
[Jupyter binder](https://mybinder.org/v2/gh/theochem/bfit/master?labpath=%2Fexamples%2)
or various files in the example [folder](https://github.com/theochem/BFit/tree/master/examples)
to see specific examples on how to fit using the different algorithms and objective
functions.
For further information about the api, please visit
[--BFit Documentation--](https://bfit.qcdevs.org/).

The instructions to access the results of the fitted atomic densities using KL-FI method is
shown in the section below.

To report any issues or ask questions, either [open an issue](
https://github.com/theochem/bfit/issues/new) or email [qcdevs@gmail.com]().

## Citation

Please use the following citation in any publication using BFit library:

```bibtex
@article{bfit2023,
author = {Tehrani, Alireza and Anderson, James S. M. and Chakraborty, Debajit and Rodriguez-Hernandez, Juan I. and Thompson, David C. and Verstraelen, Toon and Ayers, Paul W. and Heidar-Zadeh, Farnaz},
title = {An information-theoretic approach to basis-set fitting of electron densities and other non-negative functions},
journal = {Journal of Computational Chemistry},
volume = {44},
number = {25},
pages = {1998-2015},
doi = {https://doi.org/10.1002/jcc.27170},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.27170},
year = {2023}
}
```

## Dependencies

- Python >= 3.9: http://www.python.org/
- NumPy >= 1.18.5: http://www.numpy.org/
- SciPy >= 1.5.0: http://www.scipy.org/
- Matplotlib >=3.2.0: https://matplotlib.org/
- Sphinx >= 2.3.0: https://www.sphinx-doc.org/

## Installation

There are two options to install BFit:

```bash
# install from source
git clone https://github.com/theochem/bfit.git
pip install .

# or install using pip.
pip install qc-bfit

# run tests to make sure BFit was installed properly
pytest -v .
```


## Features

The features of this software are:

- Gaussian Basis set model:
  - Construct s-type and p-type Gaussian functions,
  - Compute Atomic Densities or Molecular Densities.

- Fitting measures:
  - Least-squares,
  - Kullback-Leibler divergence,
  - Tsallis divergence.

- Optimization procedures
  - Optimize using SLSQP in "scipy.minimize" procedures.
  - Optimize Kullback-Leibler using self-consistent iterative method see [paper](#citing).
  - Greedy method for optimization of Kullback-Leibler and Least-Squares, see [paper](#citing).

- Read/Parse Hatree-Fock wavefunctions for atomic systems:
  - Includes: anions, cations and heavy elements, see [data](data/README.md) page.
  - Compute:
    - Atomic density, including core, and valence densities,
    - Positive definite kinetic energy density.


## Final Models of Fitting Atomic Densities

The final model of fitting the atomic densities using the Kullback-Leibler (KL) divergence fixed point iteration method
can be accessed by opening the file `./bfit/data/kl_fpi_results.npz` with numpy.
Similarly, the results from optimizing KL with SLSQP method using `kl_fpi_results.npz`
as initial guesses can be accessed by opening the file `./bfit/data/kl_slsqp_results.npz` with numpy.
In general, we recommend KL-SLSQP results over the KL-FPI results.
```python
import numpy as np

element = "be"
results = np.load("./bfit/data/kl_fpi_results.npz")
num_s = results["be_num_s"]  # Number of s-type Gaussian function
num_p = results["be_num_p"]  # Number of p-type Gaussian functions
coeffcients = results["be_coeffs"]
exponents = results["be_exps"]

print("s-type exponents")
print(exponents[:num_s])
print("p-type exponents")
print(exponents[num_s:])
```

Alternatively, one can load these results using JSON file.
```python
import json
import numpy as np

element = "be"
with open("./bfit/data/kl_fpi_results.json") as file:
    data = json.load(file)
    data_element = data[element]

    num_s = data_element["num_s"]
    num_p = data_element["num_p"]
    coeffcients = np.array(data_element["coeffs"])
    exponents = np.array(data_element["exps"])
```

Evaluation of the normalized Gaussian model at a given set of points can also be computed
```python
from bfit.grid import ClenshawRadialGrid
from bfit.model import AtomicGaussianDensity

grid = ClenshawRadialGrid(4, num_core_pts=10000, num_diffuse_pts=899, extra_pts=[50, 75, 100])
model = AtomicGaussianDensity(grid.points, num_s=num_s, num_p=num_p, normalize=True)
model_pts = model.evaluate(coefficients, exponents)

print("Numerical integral (spherically) of the model %f." %
      grid.integrate(model_pts - 4.0 - np.pi - grid.points--2.0)
)
```

## Examples
There are four steps to using BFit.

### 1. Specify the Grid Object.
The grid is a uniform one-dimension grid with 100 points from 0. to 50.
```python
import numpy as np
from bfit.grid import UniformRadialGrid
grid = UniformRadialGrid(num_pts=100, min_radii=0., max_radii=50.)
```
See [grid.py](bfit/grid.py), for different assortment of grids.

### 2. Specify the Model Object.
Here, the model distribution is 5 s-type, normalized Gaussian functions with center at the origin.
```python
from bfit.model import AtomicGaussianDensity
model = AtomicGaussianDensity(grid.points, num_s=5, num_p=0, normalize=True)
```
See [model.py](bfit/model.py) for more options of Gaussian models.

### 3. Specify error measure.
The algorithm is fitted based on the [paper](#citing).

```python
from bfit.fit import KLDivergenceFPI

# What you want fitted to should also be defined on `grid.points`.
density = np.array([...])
fit = KLDivergenceFPI(grid, density, model)
```
See [fit.py](bfit/fit.py) for options of fitting algorithms.

### 4. Run the optimization procedure.
Initial guesses for the coefficients and exponents of the 5 s-type Gaussians must be provided.
```python
# Provide Initial Guesses
c0 = np.array([1., 1., 1., 1.])
e0 = np.array([0.001, 0.1, 1., 5., 100.])

# Optimize both coefficients and exponents and print while running.
result = fit.run(c0, e0, opt_coeffs=True, opt_expons=True, maxiter=1000, disp=True)

print("Was it successful? ", result["success"])
print("Optimized coefficients are: ", result["coeffs"])
print("Optimized exponents are: ", result["exps"])
print("Final performance measures are: ", result["fun"][-1])
```
See the [example directory](examples/) for more examples or launch the interactive binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/theochem/bfit/master?labpath=%2Fexamples%2F)
