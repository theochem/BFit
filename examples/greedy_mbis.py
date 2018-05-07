r"""
Example showing how to use greedy-fitting algorithms.
"""

from fitting.radial_grid.general_grid import RadialGrid
from fit_densities import fit_gaussian_densities
import numpy as np


grid = np.arange(0., 25, 0.25)
g_obj = RadialGrid(grid)

options = {}
fit_gaussian_densities(grid=g_obj, element_name="be", inte_val=4, method="greedy-kl_divergence",
                       iplot=True)
