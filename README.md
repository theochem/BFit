BFit <a href='https://docs.python.org/3.5/'><img src='https://img.shields.io/badge/python-3.5-blue.svg'></a>
===================

BFit is a python program that is used to fit a convex sum of 
positive basis functions to any probability distribution. 

Primarily intended for quantum chemistry community, where the basis functions are Gaussian functions and the 
fitted probability distribution is the electron density.

## Table of Contents
1. [Features](#features)
2. [Dependences](#dependences)
3. [Installation](#installation)
4. [Running Tests](#runningtests)
5. [Examples](#examples)  
6. [Questions/Issues](#questionsissues)
6. [Citing](#citing)
7. [License](#license)
8. [FAQ](#faq)  
    6.1 [Where did you get the slater coefficients from?](#where-did-you-get-the-slater-coefficients-from?)  


## Features 
The features of this software are:

* Gaussian Basis sets:
    * Handle S-type and P-type Gaussian functions.
    * Handle Atomic Densities or Molecular Densities. 
    * Handle any dimensions.

* Fitting Measures:
    * Least-squares method,
    * Kullback-Leibler method.

* Optimization Procedures
    * Optimize using "scipy.minimize" procedures.
    * Optimize Kullback-Leibler using self-consistent iterative method see [paper](#citing).

* Construct Slater atomic densities, including anions, cations and heavy elements, see [data page](data/README.md).


## Dependences 
* [Numpy](http://www.numpy.org/) 

* [Scipy](https://www.scipy.org/)

* [Matplotlib](https://matplotlib.org/)

* [Nose](http://nose.readthedocs.io/en/latest/)

* [Horton](https://theochem.github.io/horton/2.1.0/index.html) (optional)

## Installation
In your terminal run:

```bash
git clone https://github.com/QuantumElephant/fitting.git
python ./setup.py install
```

Run tests to see if it's installed properly:
```bash
nosetests -v fitting
```

## Example
There are four steps to using BFit.

### 1. Specify the Grid Object.
The grid is a uniform one-dimension grid with 100 points from 0. to 50.
```python
from bfit.grid import UniformRadialGrid
grid = UniformRadialGrid(num_pts=100, min_radii=0., max_radii=50.)
```
See [grid.py](bfit/grid.py), for different assortment of grids.

### 2. Specify the Model Object.
Here, the model distribution is 5 S-type, normalized Gaussian functions with center at the origin.
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

### 4. Run it.
```python
# Provide Initial Guesses
c0 = np.array([1., 1., 1., 1.])
e0 = np.array([0.001, 0.1, 1., 5., 100.])

# Optimize both coefficients and exponents.
result = fit.run(c0, e0, opt_coeffs=True, opt_expons=True, maxiter=1000)

print("Optimized coefficients are: ", result["x"][0])
print("Optimized exponents are: ", result["x"][1])
print("Final performance measures are: ", result["performance"][-1])
print("Was it successful? ", result["success"])
```
See the [example directory](examples/) for more examples.


## Questions/Issues
Please set up an issue or can email the following, 
- Alireza Tehrani at "alirezatehrani24@gmail.com"
- Farnaz Heidar-Zadeh at "".
- Paul W. Ayers at "". 

## Citing 
This software was written by Alireza Tehrani and Farnaz Heidar-Zadeh.

Please cite the following.
TODO: Update PAPER

Alireza Tehrani, Farnaz Heidar-Zadeh, James S.M. Anderson, Toon Verstraelen, Rogelio Cuevas-Saavedra, Ivan Vinogradov, Debajit Chakraborty, Paul W. Ayers. "BFit: Information-Theoretic Approach to Basis-Set Fitting of Electron Densities"


## License 
FittingBasisSets is distributed under the conditions of the GPL License 
version 3 (GPLv3)

## FAQ 
#### Where did you get the slater coefficients from?
Please see [Data Readme](data/) in the data folder.
