FittingBasisSets <a href='https://docs.python.org/3.5/'><img src='https://img.shields.io/badge/python-3.5-blue.svg'></a>
===================

FittingBasisSets is a python program that is used to fit a convex sum of 
positive, integrable basis functions to a probability distribution. 

Primarily used for ab-intio quantum chemistry calculations, where the basis functions of 
interest are gaussian-exponential functions and the fitted probability 
distribution is the electron density.

## Table of Contents
1. [Common usage](#commonusage)
2. [Dependences](#dependences)
3. [Installation](#installation)
4. [Running Tests](#runningtests)
5. [Examples](#examples)
6. [FAQ](#faq)  
    6.1 [Use in Quantum Chemistry?](#use-in-quantum-chemistry?)  
    6.2 [Why Gaussian Basis Sets?](#why-gaussian-basis-sets?)  
    6.3 [I want to implement a different basis set?](#how-to-implement-a-different-basis-set?)  
    6.4 [How to fit to slater densities?](#how-to-fit-to-slater-densities?)
7. [Citing](#citing)
8. [License](#license)


## Features 
    Given, ,

## Dependences 
* [Numpy](http://www.numpy.org/) 

* [Scipy](https://www.scipy.org/)

* [Matplotlib](https://matplotlib.org/)

* [Nose](http://nose.readthedocs.io/en/latest/)

* [Horton](https://theochem.github.io/horton/2.1.0/index.html) (optional)

## Installation
In your terminal run:

```python
python ./setup.py install
```

Or, if you have git installed:
```python
git clone https://github.com/Ali-Tehrani/fitting.git
```

## Running Tests 
Run tests to see if it's installed properly:
```python
nosetests -v fitting
```

## Examples

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


#### How to implement a different basis set?


## More Info

## Citing 

## License 
FittingBasisSets is distributed under the conditions of the GPL License 
version 3 (GPLv3)
