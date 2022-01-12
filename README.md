BFit
====
<a href='https://docs.python.org/3.6/'><img src='https://img.shields.io/badge/python-3.6-blue.svg'></a>
<a href='https://docs.python.org/3.7/'><img src='https://img.shields.io/badge/python-3.7-blue.svg'></a>
<a href='https://docs.python.org/3.8/'><img src='https://img.shields.io/badge/python-3.8-blue.svg'></a>
<a href='https://docs.python.org/3.9/'><img src='https://img.shields.io/badge/python-3.9-blue.svg'></a>
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/theochem/bfit/master?labpath=%2Fexamples%2F)

BFit is a Python library for (best) fitting a convex sum of positive basis functions to any
probability distribution. It is primarily intended for quantum chemistry applications, where the
basis functions are Gaussians and the fitted probability distribution is a scalar function like
the electron density.

To report any issues or ask questions, either [open an issue](
https://github.com/QuantumElephant/bfit/issues/new) or email [qcdevs@gmail.com]().


Citation
--------
Please use the following citation in any publication using BFit library:

> **"BFit: Information-Theoretic Approach to Basis-Set Fitting of Electron Densities."**,
> A. Tehrani, F. Heidar-Zadeh, J. S. M. Anderson, T. Verstraelen, R. Cuevas-Saavedra,
> I. Vinogradov, D. Chakraborty, P. W. Ayers
> `REFERENCE <https://doi.org/10.1002/jcc.26468>`__.


Dependencies
------------
* Python >= 3.0: http://www.python.org/
* NumPy >= 1.18.5: http://www.numpy.org/
* SciPy >= 1.5.0: http://www.scipy.org/
* Matplotlib >=3.2.0: https://matplotlib.org/
* Sphinx >= 2.3.0: https://www.sphinx-doc.org/


Installation
------------
Three options to install BFit:

```bash
# install from source
git clone https://github.com/theochem/bfit.git
pip install .

 # or install using conda.
conda install -c theochem qc-bfit

# or install using pip.
pip install qc-bfit

# run tests to make sure BFit was installed properly
pytest -v . 
```


Features
--------

The features of this software are:

* Gaussian Basis set model:
    * Construct s-type and p-type Gaussian functions,
    * Compute Atomic Densities or Molecular Densities. 

* Fitting measures:
    * Least-squares,
    * Kullback-Leibler divergence,
    * Tsallis divergence.

* Optimization procedures
    * Optimize using SLSQP in "scipy.minimize" procedures.
    * Optimize Kullback-Leibler using self-consistent iterative method see [paper](#citing).
    * Greedy method for optimization of Kullback-Leibler and Least-Squares, see [paper](#citing).

* Read/Parse Slater wavefunctions for atomic systems:
  * Includes: anions, cations and heavy elements, see [data](data/README.md) page.
  * Compute:
    * Atomic density,
    * Kinetic density.


## Example
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
from bfit.fit import KLDivergenceSCF

# What you want fitted to should also be defined on `grid.points`.
density = np.array([...]) 
fit = KLDivergenceSCF(grid, density, model)
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
