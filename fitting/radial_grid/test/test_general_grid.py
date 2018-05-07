r"""Test file for 'fitting.radial_grid.general_grid'."""


import numpy as np
import numpy.testing as npt
from fitting.radial_grid.general_grid import RadialGrid


def test_input_for_radial_grid():
    r"""Input checks for general grid."""
    npt.assert_raises(TypeError, RadialGrid, 5.)
    # Only one dimensional arrays are required.
    npt.assert_raises(ValueError, RadialGrid, np.array([[5.]]))


def test_integration_general_grid():
    r"""Test normal integration on the radial grid."""
    grid = np.arange(0., 2., 0.0001)
    rad_obj = RadialGrid(grid)

    # Assume no masked values.
    model = grid
    true_answer = rad_obj.integrate(model)
    desired_answer = 2. * 2. / 2.
    npt.assert_allclose(true_answer, desired_answer, rtol=1e-3)
