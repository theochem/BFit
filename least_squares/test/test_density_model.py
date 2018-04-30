
"""Test file for least_squares model."""

from fitting.least_squares import DensityModel
import numpy as np
import numpy.testing as npt

__all__ = [
    "test_residual_gives_error",
    "test_diffuse_error",
    "test_generation_ugbs_exponents",
    "test_inputs_density_model",
    "test_integration_error",
    "test_integration_model_trapz",
    "test_not_implemented"
]


def test_inputs_density_model():
    r"""Test to check inputs for 'least_squares.DensityModel'."""
    g = np.array([50.])
    e = np.array([50., 100.])
    npt.assert_raises(TypeError, DensityModel, g, 10.)
    npt.assert_raises(TypeError, DensityModel, 10., "s")
    npt.assert_raises(TypeError, DensityModel, g, "s", 10.)
    npt.assert_raises(ValueError, DensityModel, g, "s", e)
    assert DensityModel(g).element is None
    dens_obj = DensityModel(g, "S")
    npt.assert_equal(dens_obj.grid, g)
    assert dens_obj.element == "s"


def test_not_implemented():
    r"""Tests for not implemeneted for least_squares."""
    d = DensityModel(np.array([5.]), "S")
    npt.assert_raises(NotImplementedError, d.cost_function)
    npt.assert_raises(NotImplementedError, d.create_model)
    npt.assert_raises(NotImplementedError,
                      d.derivative_of_cost_function)
    npt.assert_raises(NotImplementedError, d.create_cofactor_matrix)


def test_integration_model_trapz():
    r"""Test integration method for 'least_squares.DensityModel'."""
    # Test integration of r^2 e^(-r^2) dr
    grid = np.arange(0, 25, 0.25)
    dens_obj = DensityModel(grid)
    func_vals = np.exp(-grid**2)
    actual_value = dens_obj.integrate_model_trapz(func_vals)
    desired_val = 0.443313
    assert np.abs(actual_value - desired_val) < 0.1


def test_diffuse_error():
    r"""Test error measure for 'least_squares.DensityModel.'"""
    # Test if integration gives zero if they're the same
    grid = np.arange(0, 25, 0.25)
    fun_vals = np.exp(-grid**2)
    dens_obj = DensityModel(grid, electron_density=fun_vals)
    actual_value = dens_obj.get_error_diffuse(fun_vals, fun_vals)
    assert actual_value < 1e-5


def test_integration_error():
    r"""Test integration error measure for 'least_squares.DensityModel.'"""
    # Test if integration gives zero if they're the same
    grid = np.arange(0, 25, 0.25)
    fun_vals = np.exp(-grid ** 2)
    dens_obj = DensityModel(grid, electron_density=fun_vals)
    actual_value = dens_obj.get_integration_error(fun_vals, fun_vals)
    assert actual_value < 1e-5

    # Test linearity of integration
    fun_vals2 = 5. * np.exp(-grid**2.)
    actual_value2 = dens_obj.get_integration_error(fun_vals, fun_vals2)
    assert actual_value2 - 5. * 0.443313 < 1e-3


def test_residual_gives_error():
    r"""Test residual gives NotImplementedError for 'least_squares.DensityModel."""
    grid = np.arange(0, 25, 0.25)
    fun_vals = np.exp(-grid ** 2)
    dens_obj = DensityModel(grid, electron_density=fun_vals)
    npt.assert_raises(NotImplementedError, dens_obj.get_residual)


def test_generation_ugbs_exponents():
    r""""""
    pass
