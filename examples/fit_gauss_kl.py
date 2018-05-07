r"""
Example showing how to fit to any distribution using kullback-leibler divergence.
"""

from radial_grid.radial_grid import RadialGrid
from fit_densities import fit_gaussian_densities
import numpy as np
import matplotlib.pyplot as plt

# Specifying a uniform grid.
grid = np.arange(0., 6, 0.25)
grid_obj = RadialGrid(grid)

# Model An Arbritary Probability Density
true_model = np.exp(-3 * grid)

# Run Kullback-Leibler Divergence Optimization With Initial Guesses
options = {"coeffs": np.arange(1, 20, dtype=float),
           "fparams": np.arange(1, 20, dtype=float),
           "eps_coeff": 1e-3, "eps_fparam": 1e-4, 'iprint': True}
params = fit_gaussian_densities(grid=grid_obj, true_model=true_model, method="kl_divergence",
                                options=options)


# Run Least-Squares Using SLSQP
initial_guess = np.append(np.arange(1, 20, dtype=float),
                          np.arange(1, 20, dtype=float))
options = {"initial_guess": initial_guess}
params_lq = fit_gaussian_densities(grid=grid_obj, true_model=true_model, method="slsqp",
                                   options=options)


# Plotting The Results
from fitting.kl_divergence.gaussian_kl import GaussianKullbackLeibler
gkl = GaussianKullbackLeibler(grid_obj, true_model)
model_kl = gkl.get_model(params[:len(params)//2], params[len(params)//2:])
model_lq = gkl.get_model(params_lq[:len(params) // 2], params_lq[len(params) // 2:],
                         norm=False)

plt.plot(grid, model_kl, "bo-", label="KL Divergence Model")
plt.plot(grid, true_model, "ro-", label="True Model")
plt.plot(grid, model_lq, "go-", label="Least Squares Model")
plt.legend()
plt.show()
