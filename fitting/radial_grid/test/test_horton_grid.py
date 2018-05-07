
"""Test file for horton_grid."""

from fitting.radial_grid.horton import HortonGrid
import numpy as np
import numpy.testing as npt

__all__ = [
    "test_input_checks_horton_grid",
    "test_integration_on_horton_grid"
]

np.random.seed(101010101)


def test_input_checks_horton_grid():
    r"""Test input checks on horton _grid."""
    npt.assert_raises(TypeError, HortonGrid, "not numb", 1, 1)
    npt.assert_raises(TypeError, HortonGrid, 1, "not numb", 1)
    npt.assert_raises(TypeError, HortonGrid, 1, 1, 2.1)
    npt.assert_raises(ValueError, HortonGrid, 1, 1, -1)


def test_integration_on_horton_grid():
    r"""Test integration on horton _grid."""
    numb_pts = 100
    rad_obj = HortonGrid(0., 25, numb_pts)
    arr = np.exp(-rad_obj.radii ** 2)
    actual_value = rad_obj.integrate(arr)
    desired_val = 4. * np.pi * 0.443313
    assert np.abs(actual_value - desired_val) < 0.1
