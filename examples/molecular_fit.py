r"""
This file shows how to do a molecular fitting based on a water format checkpoint file (.fchk).
"""

from fitting.kl_divergence.molecular_fitting import MolecularFitting
from fitting.radial_grid.cubic_grid import CubicGrid
from chemtools import HortonMolecule
import numpy as np


# Define Initial guesses for gaussian coefficients and exponents water.
# Spread out exponents seems to be a a proper answer.
coeffs_h2o = np.array([10.] * 21)
exps_h2o = np.array([0.01, 0.1, 1, 10, 50, 100, 1000,
                     0.01, 0.1, 1, 10, 50, 100, 1000,
                     0.01, 0.1, 1, 10, 50, 100, 1000])
number_of_parameters = [7, 7, 7]

# Define the grid and create the CubicGrid Object
grid = []
step_size = 0.05
for x in np.arange(-3.2, 3.2 + step_size, step_size):
    for y in np.arange(-3.2, 3.2 + step_size, step_size):
        for z in np.arange(-3.2, 3.2 + step_size, step_size):
            grid.append([x, y, z])
grid_obj = CubicGrid(np.array(grid), step_size)


# Create Molecular Density Values from fchk file.
molecule_file = "./examples/h2o_holy.fchk"
molecule = HortonMolecule.from_file(molecule_file)
dens_val = molecule.compute_density(grid_obj.grid)
print("Integration of Entire Density", np.sum(dens_val) * step_size**3.)

# Create Molecular Fitting object and run it
mole_fit = MolecularFitting(grid_obj, dens_val=dens_val, inte_val=10.,
                            mol_coords=molecule.coordinates,
                            number_of_params=number_of_parameters)

mole_fit.run(1e-2, 1e-3, coeffs_h2o, exps_h2o, True)
"""array([7.32582208e-02, 2.09348586e+00, 4.36255687e+00, 5.40025011e-01,
       2.65198449e-01, 1.09926268e+00, 4.40488090e-01, 2.15809445e-02,
       1.13217309e-03, 9.01662939e-02, 2.10606756e-01, 1.49706255e-01,
       8.69445386e-03, 1.23586550e-03, 8.91004971e-04, 1.14887463e-05,
       8.21590995e-05, 9.01662939e-02, 2.65895129e-01, 2.64568791e-01,
       1.69137127e-02, 2.24128868e-03, 1.57117916e-03, 2.61036585e-04]),
        array([3.76950623e-01, 3.78621126e-01, 1.18199322e+00, 3.14849867e+00,
       1.59452957e+01, 3.63345589e+01, 1.24449823e+02, 3.18027127e+03,
       3.76950869e-01, 6.32680860e-01, 1.42001413e+00, 3.20942115e+00,
       1.19297492e+01, 2.92684717e+01, 3.60627118e+01, 1.87479052e+03,
       6.32575373e-01, 6.32680860e-01, 7.30231466e-01, 2.38091718e+00,
       8.62620915e+00, 1.80712389e+01, 2.33616364e+01, 7.89299853e+01])
       7, 7, 7"""
