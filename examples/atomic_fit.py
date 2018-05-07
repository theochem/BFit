r"""
Example file to show how to use fit_densities.fit_radial_densities.

This example shows how to fit to a slater densities specified by the
element name (here it is beryllium) rather than providing a slater density.
The fit_densities handles the work of constructing a slater density for you.
"""

from radial_grid.clenshaw_curtis import ClenshawGrid
from fit_densities import fit_gaussian_densities
import numpy as np


# Fitting to Beryllium Slater Density

# First Define the grid. Here we are using clenshaw-curtis grid.
atomic_number = 4
numb_pts = 200
extra_pts = [25, 50, 75]
g_obj = ClenshawGrid(atomic_number, numb_pts, numb_pts, extra_pts)

# Arguments that Kullback-Leibler Divergence takes.
# Initial Guess are:
coeff = np.array([1., 2., 3., 4., 5.])
exps = np.array([1., 2., 3., 4., 5.])
options = {"eps_coeff": 1e-5, "eps_fparam": 1e-6, "coeffs": coeff,
           "fparams": exps, 'iprint': True}
opt_params = fit_gaussian_densities(grid=g_obj, element_name="be", inte_val=atomic_number,
                                    method="kl_divergence", options=options)
print("Updated Parameters:")
coeffs = opt_params[:len(opt_params)//2]
exps = opt_params[len(opt_params)//2:]
