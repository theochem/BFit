FittingBasisSets <a href='https://docs.python.org/3.5/'><img src='https://img.shields.io/badge/python-3.5-blue.svg'></a>
===================

FittingBasisSets is a python program that is used to fit a convex sum of 
positive basis functions to any probability distribution. 

Primarily used for ab-intio quantum chemistry calculations, where the basis functions of 
interest are gaussian-exponential functions and the fitted probability 
distribution is the electron density.

## Table of Contents
1. [Features](#features)
2. [Dependences](#dependences)
3. [Installation](#installation)
4. [Running Tests](#runningtests)
5. [Examples](#examples)  
    5.1 [Fit Gaussian set to a specified atom (slater basis set)](#fit-gaussian-set-to-your-own-function.)  
    5.2 [Fit gaussian set to your own function.](#fit-gaussian-set-to-your-own-function.)  
    5.3 [Fit gaussian set to molecular density instead of atomic density.](#fit-gaussian-set-to-molecular-density-instead-of-atomic-density.)
6. [FAQ](#faq)  
    6.1 [Why Gaussian Basis Sets?](#why-gaussian-basis-sets?)  
    6.2 [Where did you get the slater coefficients from?](#where-did-you-get-the-slater-coefficients-from?)  
    6.3 [I want to implement a different basis set?](#how-to-implement-a-different-basis-set?)  
7. [Citing](#citing)
8. [License](#license)


## Features 
* Fitting Radial Gaussian Basis sets to Atomic Densities

* Fitting Gaussian Basis sets to Molecular Densities

* Fitting Valence Gaussian Basis Sets to Atomic Densities

* Optimize Least-squares of Radial Gaussian Basis Sets via SLSQP

* Optimize Kullback-Leibler of Radial Gaussian Sets via SLSQP


## Dependences 
* [Numpy](http://www.numpy.org/) 

* [Scipy](https://www.scipy.org/)

* [Matplotlib](https://matplotlib.org/)

* [Nose](http://nose.readthedocs.io/en/latest/)

* [Horton](https://theochem.github.io/horton/2.1.0/index.html) (optional)

## Installation
In your terminal run:

```python
git clone https://github.com/Ali-Tehrani/fitting.git
python ./setup.py install
```

Or, if you have pre-requisites already installed:
```python
git clone https://github.com/Ali-Tehrani/fitting.git
```

## Running Tests 
Run tests to see if it's installed properly:
```python
nosetests -v fitting
```

## Examples
We assume the gaussian basis set is normalized.
#### Fit Gaussian set to a specified atom (slater basis set).
Please see the [atomic_fit.py](examples/atomic_fit.py) file located in the example folder.

#### Fit gaussian set to your own function.
Please see [fit_gauss_kl.py](examples/fit_gauss_kl.py) file located in the 
example folder.

#### Fit gaussian set to molecular density instead of atomic density.
Please see [molecular_fit.py](examples/molecular_fit.py) file located in the 
example folder.



## FAQ 
#### Why Gaussian Basis Sets?
The basis sets of importance are, slater densities and gaussian densities.
Slater densities are more accurate for modelling atoms, due to the cusp at r=0, 
however are 
numerically harder to integrate. On the other hand,
gaussian basis sets are not accurate for modeling atoms, due to the lack of cusp
 at r=0, 
however 
are easier to integrate.
Curve-fitting procedures is used to convert between the two however modeling
this problem as a standard least-squares does not suffice to solve it, 
instead a different objective function is used to solve this problem.


#### Where did you get the slater coefficients from?
Please see [Data Readme](data/) in the data folder.


#### How to implement a different basis set?
If you want to fit with a different basis set. You will first need
to write down the formulas for how to update coefficients and 
function parameters. Please see the paper below.

Next you need to create a class inside your own python file, with a parent
class of [KullbackLeiblerFitting](fitting/kl_divergence/kull_leib_fitting.py),
implementing the abstract methods provided. In other words,
```python
class YourOwnBasisSet(KullbackLeiblerFitting):
    def __init__(self, grid_obj, true_model, ...):
        ...
        super(YourOwnBasisSet, self).__init__(grid_obj, true_model, ...)
    
    # Implement the abstract methods:
    def get_model(self, coeffs, fparams):
        return your model
        
    def _update_coeffs(self, coeffs, fparams):
        Update your coefficients 
        ...
        
    def _update_fparams(self, coeffs, fparams):
        Update your function parameters
        ...
        
    def _get_norm_constant(self, coeffs, fparams):
        Return normalization constant for one basis function.
        ...
```
Afterwards, you'll need to run it,
```python
obj = YourOwnBasisSet(your parameters)
new_params = obj.run()
print("Updated Parameters ", new_params)
```

## More Info
Please email, whomever. 

## Citing 

## License 
FittingBasisSets is distributed under the conditions of the GPL License 
version 3 (GPLv3)
