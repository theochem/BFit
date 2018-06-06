r"""
This file shows how to fit using Kullback-leibler with weights added.
"""
import numpy as np
from fitting.grid import BaseRadialGrid
from fitting.density import AtomicDensity
from fitting.kl_divergence.gaussian_kl import GaussianKullbackLeibler

# First Define the grid. Here we are using clenshaw-curtis grid.
grid = np.arange(0.001, 25., 0.001)
grid_obj = BaseRadialGrid(grid)

# Define what you want to fit. Here we want slater densities of beryllium.
true_model = AtomicDensity("./data/examples/be.slater", grid).electron_density
integration_value = 4.

# Define your weights.
weights = 1. / (4. * np.pi * grid_obj.radii**2.)

# Define initial guess.
coeffs = np.array([4. / 10.] * 10)
exponents = np.array([0.001, 0.01, 0.1, 0.5, 1., 10., 50., 100., 500., 1000.])


# Define the GaussianKL object and run it with accuracies around 1e-3.
gauss_obj = GaussianKullbackLeibler(grid_obj, true_model, integration_value, weights)
gauss_obj.run(1e-3, 1e-4, coeffs, exponents, iprint=True)
