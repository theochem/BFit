
"""Test file for radial _grid."""

from fitting.radial_grid import ClenshawGrid, HortonGrid
import numpy as np
import numpy.testing as npt

__all__ = [
    "test_grid_is_clenshaw",
    "test_grid_is_uniform",
    "test_input_checks_horton_grid",
    "test_input_checks_radial_grid",
    "test_integration_on_grid"
]

np.random.seed(101010101)


def test_input_checks_radial_grid():
    r"""Test input checks on radial _grid."""
    npt.assert_raises(TypeError, ClenshawGrid, 10.1, 1, 1, [])
    npt.assert_raises(ValueError, ClenshawGrid, -10, 1, 1, [])
    npt.assert_raises(TypeError, ClenshawGrid, 1, 2.2, 1, [])
    npt.assert_raises(ValueError, ClenshawGrid, 1, -2, 1, [])
    npt.assert_raises(TypeError, ClenshawGrid, 1, 1, 1.1, [])
    npt.assert_raises(ValueError, ClenshawGrid, 1, 1, -2, [])
    npt.assert_raises(TypeError, ClenshawGrid, 1, 1, 1, "not list")


def test_grid_is_clenshaw():
    r"""Test that radial _grid returns a clenshaw _grid."""
    core_pts = 10
    diff_pts = 20
    atomic_numb = 10
    fac = 1. / (2. * 10.)
    rad_obj = ClenshawGrid(atomic_numb, core_pts, diff_pts, [1000])
    actual_pts = rad_obj.radii
    desired_pts = []
    for x in range(0, core_pts):
        desired_pts.append(fac * (1. - np.cos(np.pi * x / (2. * core_pts))))
    for x in range(1, diff_pts):
        desired_pts.append(25. * (1. - np.cos(np.pi * x / (2. * diff_pts))))
    desired_pts.append(1000)
    desired_pts = np.sort(desired_pts)
    npt.assert_allclose(actual_pts, desired_pts)


def test_grid_is_uniform():
    r"""Test that radial _grid returns a uniform _grid."""
    numb_pts = 10
    actual_pts = ClenshawGrid.uniform_grid(numb_pts)
    desired_pts = [x / numb_pts for x in range(0, 100 * numb_pts)]
    npt.assert_allclose(actual_pts, desired_pts)


def test_integration_on_grid():
    r"""Test that integrations works on radial _grid."""
    # Test exponential with wolfram
    numb_pts = 100
    rad_obj = ClenshawGrid(10, numb_pts, numb_pts)
    arr = np.exp(-rad_obj.radii**2)
    actual_value = rad_obj.integrate_spher(arr)
    desired_val = 4. * np.pi * 0.443313
    assert np.abs(actual_value - desired_val) < 0.1

    # Test with singularities.
    numb_pts = 100
    rad_obj = ClenshawGrid(10, numb_pts, numb_pts, filled=True)
    arr = np.exp(-rad_obj.radii ** 2)
    arr[np.random.randint(5)] = np.nan
    actual_value = rad_obj.integrate_spher(arr)
    desired_val = 4. * np.pi * 0.443313
    assert np.abs(actual_value - desired_val) < 0.1


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
