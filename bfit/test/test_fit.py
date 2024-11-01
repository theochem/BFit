# -*- coding: utf-8 -*-
# BFit is a Python library for fitting a convex sum of Gaussian
# functions to any probability distribution
#
# Copyright (C) 2020- The QC-Devs Community
#
# This file is part of BFit.
#
# BFit is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# BFit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# ---
r"""Test bfit.fit module."""

from bfit.fit import KLDivergenceFPI, ScipyFit
from bfit.grid import CubicGrid, UniformRadialGrid
from bfit.measure import KLDivergence, SquaredDifference
from bfit.model import AtomicGaussianDensity, MolecularGaussianDensity
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises


def test_lagrange_multiplier():
    r"""Test lagrange multiplier of KL-SCF."""
    g = UniformRadialGrid(150, 1e-4, 15.0)
    e = np.exp(-g.points)
    kl = KLDivergenceFPI(g, e, None, spherical=False)
    assert_almost_equal(kl.lagrange_multiplier, 1., decimal=8)


def test_goodness_of_fit():
    r"""Test goodness of fit with analytic values."""
    g = UniformRadialGrid(1000, 0.0, 10.0)
    e = np.exp(-g.points)
    m = AtomicGaussianDensity(g.points, num_s=1, num_p=0, normalize=False)
    kl = KLDivergenceFPI(g, e, m, mask_value=0., spherical=True)
    gf = kl.goodness_of_fit(np.array([1.]), np.array([1.]))
    expected = [
        5.56833, 4 * np.pi * 1.60909, 0.128, 4.0 * np.pi * 0.0882922, 4. * np.pi * 17.360
    ]
    assert_almost_equal(expected, gf, decimal=1)


def test_assertion_raises():
    r"""Test assertion raises of all fitting methods."""
    g = UniformRadialGrid(1000, 0.0, 10.0)
    e = np.exp(-g.points)
    m = AtomicGaussianDensity(g.points, num_s=1, num_p=0, normalize=False)
    kl = KLDivergenceFPI(g, e, m, mask_value=0., spherical=True)
    assert_raises(ValueError, kl._update_params, None, None, False, False)
    assert_raises(ValueError, kl.run, np.array([1., 2.]), np.array([1.]))
    assert_raises(ValueError, kl.run, np.array([1.]), np.array([1., 2.]))
    assert_raises(ValueError, kl.run, np.array([1.]), np.array([1.]), False, False)
    assert not kl.run(np.array([1.]), np.array([1.]), maxiter=1)["success"]

    m2 = AtomicGaussianDensity(g.points * 2., num_s=1, num_p=0, normalize=False)
    measure = KLDivergence()
    assert_raises(ValueError, ScipyFit, g, e, m2)
    assert_raises(ValueError, ScipyFit, g, np.append(e, 2.), m)
    assert_raises(ValueError, ScipyFit, g, e, m, measure, "non slsqp")
    assert_raises(TypeError, ScipyFit, g, e, m, "not kl")
    gb = ScipyFit(g, e, m, measure=measure)
    assert_raises(ValueError, gb.run, [], [], False, False)
    assert_raises(ValueError, gb.evaluate_model, [], ("not fixed", 2))

    # Test giving a grid class with no points returns error
    class GridNoPoints:
        def integrate(self):
            pass
    assert_raises(AttributeError, ScipyFit, GridNoPoints(), e, KLDivergence())

    # Test giving grid class with no integrate method returns error.
    class GridNoIntegrate:
        def __init__(self):
            self.points = g.points
    assert_raises(AttributeError, ScipyFit, GridNoIntegrate(), e, KLDivergence())


def test_run_normalized_s_gaussian():
    r"""Test KL-SCF against normalized s-type Gaussian."""
    # density is normalized 1s orbital with exponent=1.0
    g = UniformRadialGrid(150, 0.0, 15.0)
    e = (1. / np.pi)**1.5 * np.exp(-g.points**2.)
    model = AtomicGaussianDensity(g.points, num_s=1, num_p=0, normalize=True)
    kl = KLDivergenceFPI(g, e, model, spherical=True)

    # fit density with initial coeff=1. & expon=1.
    res = kl.run(np.array([1.]), np.array([1.]), True, True, 500, 1.e-4, 1.e-4, 1.e-4)
    # check optimized coeffs & expons
    assert_almost_equal(np.array([1.]), res["coeffs"], decimal=8)
    assert_almost_equal(np.array([1.]), res["exps"], decimal=8)
    # check value of optimized objective function & fitness measure
    assert_almost_equal(0., res["fun"][-1], decimal=8)
    assert_almost_equal(1., res["performance"][-1, 0], decimal=8)
    assert_almost_equal(0., res["performance"][-1, 1:], decimal=8)

    # fit density with initial coeff=0.5 & expon=0.5
    res = kl.run(np.array([0.5]), np.array([0.5]), True, True, 500, 1.e-4, 1.e-4, 1.e-4)
    # check optimized coeffs & expons
    assert_almost_equal(np.array([1.]), res["coeffs"], decimal=8)
    assert_almost_equal(np.array([1.]), res["exps"], decimal=8)
    # check value of optimized objective function & fitness measure
    assert_almost_equal(0., res["fun"][-1], decimal=8)
    assert_almost_equal(1., res["performance"][-1, 0], decimal=8)
    assert_almost_equal(0., res["performance"][-1, 1:], decimal=8)

    # fit density with initial coeff=0.1 & expon=10.
    res = kl.run(np.array([0.1]), np.array([10.]), True, True, 500, 1.e-4, 1.e-4, 1.e-4)
    # check optimized coeffs & expons
    assert_almost_equal(np.array([1.]), res["coeffs"], decimal=8)
    assert_almost_equal(np.array([1.]), res["exps"], decimal=8)
    # check value of optimized objective function
    assert_almost_equal(0., res["fun"][-1], decimal=8)

    # fit density with initial coeff=20. & expon=0.01.
    res = kl.run(np.array([20.]), np.array([0.01]), True, True, 500, 1.e-4, 1.e-4, 1.e-4)
    # check optimized coeffs & expons
    assert_almost_equal(np.array([1.]), res["coeffs"], decimal=8)
    assert_almost_equal(np.array([1.]), res["exps"], decimal=8)
    # check value of optimized objective function
    assert_almost_equal(0., res["fun"][-1], decimal=8)


def test_kl_scf_update_coeffs_2s_gaussian():
    r"""Test KL-SCF method for updating coefficients of two s-type Gaussians."""
    # actual density is a 1s Slater function
    grid = UniformRadialGrid(150, 0.0, 15.0)
    c, e = np.array([5., 2.]), np.array([10., 3.])
    dens = np.exp(-grid.points, dtype=np.float64)
    # model density is a normalized 2s Gaussian basis
    model = AtomicGaussianDensity(grid.points, num_s=2, num_p=0, normalize=True)
    # test updating coeffs
    kl = KLDivergenceFPI(grid, dens, model, spherical=False)
    new_coeffs, new_expons = kl._update_params(c, e, True, False)
    # compute model density
    approx = c[0] * (e[0] / np.pi)**1.5 * np.exp(-e[0] * grid.points**2)
    approx += c[1] * (e[1] / np.pi)**1.5 * np.exp(-e[1] * grid.points**2)
    # compute expected coeffs
    coeffs = c * (e / np.pi)**1.5
    ratio = dens / approx
    coeffs[0] *= grid.integrate(ratio * np.exp(-e[0] * grid.points**2))
    coeffs[1] *= grid.integrate(ratio * np.exp(-e[1] * grid.points**2))
    assert_almost_equal(new_expons, e, decimal=6)
    assert_almost_equal(new_coeffs, coeffs, decimal=6)


def test_kl_scf_update_params_2s_gaussian():
    r"""Test KL-SCF method for updating exponents of two s-type Gaussians."""
    # actual density is a 1s Slater function
    grid = UniformRadialGrid(1000, 0.0, 20.0)
    points = grid.points
    spherical = 4.0 * np.pi * points**2.0
    c, e = np.array([5., 2.], dtype=np.float64), np.array([10., 3.], dtype=np.float64)
    dens = np.exp(-points, dtype=np.float64)

    # model density is a normalized 2s Gaussian basis
    model = AtomicGaussianDensity(points, num_s=2, num_p=0, normalize=True)
    # test updating coeffs
    kl = KLDivergenceFPI(grid, dens, model, spherical=True)
    new_coeffs, new_expons = kl._update_params(c, e, False, True)
    # compute model density
    approx = c[0] * (e[0] / np.pi)**1.5 * np.exp(-e[0] * points**2)
    approx += c[1] * (e[1] / np.pi)**1.5 * np.exp(-e[1] * points**2)
    # compute expected expons
    expons = 1.5 * np.ones(2)
    ratio = np.ma.filled(dens / np.ma.array(approx), 0.)
    expons[0] *= grid.integrate(np.exp(-e[0] * points**2) * ratio * spherical)
    expons[1] *= grid.integrate(np.exp(-e[1] * points**2) * ratio * spherical)
    expons[0] /= grid.integrate((np.exp(-e[0] * points**2) * points**2) * ratio * spherical)
    expons[1] /= grid.integrate((np.exp(-e[1] * points**2) * points**2) * ratio * spherical)
    assert_almost_equal(new_coeffs, c, decimal=6)
    assert_almost_equal(new_expons, expons, decimal=6)


def test_kl_scf_update_params_1s1p_gaussian():
    r"""Test KL-SCF method against one s-type and one p-type Gaussians."""
    # actual density is a 1s Slater function
    grid = UniformRadialGrid(150, 0.0, 15.0)
    points = grid.points
    spherical = 4.0 * np.pi * points**2.0
    c, e = np.array([1., 2.]), np.array([3., 4.])
    dens = np.exp(-grid.points)
    # model density is a normalized 2s Gaussian basis
    model = AtomicGaussianDensity(points, num_s=1, num_p=1, normalize=True)
    # compute model density
    approx = c[0] * (e[0] / np.pi)**1.5 * np.exp(-e[0] * points**2)
    approx += c[1] * 2. * e[1]**2.5 * points**2 * np.exp(-e[1] * points**2) / (3 * np.pi**1.5)
    # check model.evaluate
    assert_almost_equal(approx, model.evaluate(c, e), decimal=6)
    # test updating coeffs
    kl = KLDivergenceFPI(grid, dens, model, mask_value=0., spherical=True)
    new_coeffs, new_expons = kl._update_params(c, e, update_coeffs=True, update_expons=False)
    coeffs = c * np.array([(e[0] / np.pi)**1.5, 2 * e[1]**2.5 / (3 * np.pi**1.5)])
    coeffs[0] *= grid.integrate(dens * np.exp(-e[0] * points**2) / approx * spherical)
    coeffs[1] *= grid.integrate(dens * np.exp(-e[1] * points**2) * points**2 / approx * spherical)
    assert_almost_equal(new_expons, e, decimal=6)
    assert_almost_equal(new_coeffs, coeffs, decimal=6)
    # test updating expons
    new_coeffs, new_expons = kl._update_params(c, e, update_coeffs=False, update_expons=True)
    expons = np.array([1.5, 2.5])
    expons[0] *= grid.integrate(dens * np.exp(-e[0] * points**2) / approx * spherical)
    expons[1] *= grid.integrate(dens * points**2 * np.exp(-e[1] * points**2) / approx * spherical)
    expons[0] /= grid.integrate(dens * points**2 * np.exp(-e[0] * points**2) / approx * spherical)
    expons[1] /= grid.integrate(dens * points**4 * np.exp(-e[1] * points**2) / approx * spherical)
    assert_almost_equal(new_coeffs, c, decimal=6)
    assert_almost_equal(new_expons, expons, decimal=6)
    # test updating coeffs & expons
    new_coeffs, new_expons = kl._update_params(c, e, update_coeffs=True, update_expons=True)
    assert_almost_equal(new_coeffs, coeffs, decimal=6)
    assert_almost_equal(new_expons, expons, decimal=6)


def test_kl_scf_update_params_3d_molecular_dens_1s_1s_gaussian():
    r"""Test KL-SCF method in three dimensions for one s-type and one p-type Gaussian."""
    # actual density is a 1s Gaussian at origin
    axes = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    grid = CubicGrid(np.array([-2.0, -2.0, -2.0]), axes, (40, 40, 40))
    dens = np.exp(-np.sum(grid.points ** 2., axis=1))
    # model density is a normalized 1s Gassuain on each center
    coord = np.array([[0., 0., 0.], [0., 0., 1.]])
    c, e = np.array([1., 2.]), np.array([3., 4.])
    model = MolecularGaussianDensity(grid.points, coord, np.array([[1, 0], [1, 0]]), True)
    kl = KLDivergenceFPI(grid, dens, model)
    # compute expected updated coeffs
    dist1 = np.sum((grid.points - coord[0])**2, axis=1)
    dist2 = np.sum((grid.points - coord[1])**2, axis=1)
    approx = c[0] * (e[0] / np.pi)**1.5 * np.exp(-e[0] * dist1)
    approx += c[1] * (e[1] / np.pi)**1.5 * np.exp(-e[1] * dist2)
    expected_coeffs = c * (e / np.pi)**1.5
    expected_coeffs[0] *= grid.integrate(dens * np.exp(-e[0] * dist1) / approx)
    expected_coeffs[1] *= grid.integrate(dens * np.exp(-e[1] * dist2) / approx)
    # check updated coeffs
    coeffs, expons = kl._update_params(c, e, update_coeffs=True, update_expons=False)
    assert_almost_equal(expons, e, decimal=6)
    assert_almost_equal(coeffs, expected_coeffs, decimal=6)
    # compute expected updated expons
    expected_expons = 1.5 * (e / np.pi)**1.5
    expected_expons[0] *= grid.integrate(dens * np.exp(-e[0] * dist1) / approx)
    expected_expons[1] *= grid.integrate(dens * np.exp(-e[1] * dist2) / approx)
    denoms = (e / np.pi)**1.5
    denoms[0] *= grid.integrate(dens * dist1 * np.exp(-e[0] * dist1) / approx)
    denoms[1] *= grid.integrate(dens * dist2 * np.exp(-e[1] * dist2) / approx)
    expected_expons /= denoms
    # check updated expons
    coeffs, expons = kl._update_params(c, e, update_coeffs=False, update_expons=True)
    assert_almost_equal(coeffs, c, decimal=6)
    assert_almost_equal(expons, expected_expons, decimal=6)


def test_kl_scf_run_3d_molecular_dens_1s_1p_gaussian():
    r"""Test KL-SCF on 3D Gaussian example with 1 s-type and 1 p-type Gaussians."""
    # make cubic grid
    axes = np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]])
    grid = CubicGrid(np.array([-10.0, -10.0, -10.0]), axes, (65, 65, 65))
    # actual density is a 1s Gaussian at x=0.5 & 1p Gaussian at y=0.25
    c, e = np.array([0.53, 2.07]), np.array([0.67, 1.92])
    coord = np.array([[0.5, 0., 0.], [0., 0.25, 0.]])
    dist1 = np.sum((grid.points - coord[0])**2., axis=1)
    dist2 = np.sum((grid.points - coord[1])**2., axis=1)
    dens1 = c[0] * (e[0] / np.pi)**1.5 * np.exp(-e[0] * dist1)
    dens2 = c[1] * 2. * e[1]**2.5 * dist2 * np.exp(-e[1] * dist2) / (3. * np.pi**1.5)
    # model density is a normalized 1s Gaussian on each center
    model = MolecularGaussianDensity(grid.points, coord, np.array([[1, 0], [0, 1]]), True)
    # check model norm
    assert_almost_equal(grid.integrate(model.evaluate(c, e, False)), np.sum(c), decimal=8)
    # initial coeffs & expons for optimization
    cs0, es0 = np.array([0.25, 0.01]), np.array([0.5, 2.2])
    # optimize coeffs
    kl = KLDivergenceFPI(grid, dens1 + dens2, model, mask_value=0.)
    result = kl.run(cs0, e, True, False, 100, 1.e-8, 1.e-8, 1e-15)
    assert_equal(result["success"], True)
    assert_almost_equal(result["coeffs"], c, decimal=6)
    assert_almost_equal(result["exps"], e, decimal=6)
    assert_almost_equal(result["fun"][-1], 0., decimal=6)
    # optimize expons
    result = kl.run(c, es0, False, True, 100, 1.e-8, 1.e-8, 1.e-15)
    assert_equal(result["success"], True)
    assert_almost_equal(result["coeffs"], c, decimal=6)
    assert_almost_equal(result["exps"], e, decimal=6)
    assert_almost_equal(result["fun"][-1], 0., decimal=6)
    # optimize coeffs & expons
    result = kl.run(cs0, es0, True, True, 150, 1.e-7, 1.e-7, 1.e-15)
    assert_equal(result["success"], True)
    assert_almost_equal(result["coeffs"], c, decimal=6)
    assert_almost_equal(result["exps"], e, decimal=6)
    assert_almost_equal(result["fun"][-1], 0., decimal=6)


def test_kl_fit_unnormalized_dens_normalized_1s_gaussian():
    r"""Test ScipyFit against normalized gaussian model."""
    # density is normalized 1s orbital with exponent=1.0
    grid = UniformRadialGrid(150, 0.0, 15.0)
    # density is normalized 1s gaussian
    dens = 1.57 * np.exp(-0.51 * grid.points**2.)
    model = AtomicGaussianDensity(grid.points, num_s=1, num_p=0, normalize=True)
    measure = KLDivergence()
    kl = ScipyFit(grid, dens, model, measure=measure, method="slsqp", spherical=True)
    # expected coeffs & expons
    expected_cs = np.array([1.57 / (0.51 / np.pi)**1.5])
    expected_es = np.array([0.51])
    # initial coeff=1.57 & expon=0.51
    result = kl.run(np.array([1.57]), np.array([0.51]), True, True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=8)
    assert_almost_equal(expected_es, result["exps"], decimal=8)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., 0.]), result["jacobian"], decimal=6)
    # initial coeff=0.1 & expon=0.1
    result = kl.run(np.array([0.1]), np.array([0.1]), True, True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=8)
    assert_almost_equal(expected_es, result["exps"], decimal=8)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., 0.]), result["jacobian"], decimal=6)
    # initial coeff=5.0 & expon=15.
    result = kl.run(np.array([5.0]), np.array([15.]), True, True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=8)
    assert_almost_equal(expected_es, result["exps"], decimal=8)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., 0.]), result["jacobian"], decimal=6)
    # initial coeff=0.8 & expon=0.51, opt coeffs
    result = kl.run(np.array([0.5]), np.array([0.51]), True, False)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=8)
    assert_almost_equal(expected_es, result["exps"], decimal=8)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1.]), result["jacobian"], decimal=6)


def test_kl_fit_normalized_dens_unnormalized_1s_gaussian():
    r"""Test ScipyFit against un-normalized Gaussian model."""
    # density is normalized 1s gaussian
    grid = UniformRadialGrid(200, 0.0, 15.0)
    dens = 2.06 * (0.88 / np.pi)**1.5 * np.exp(-0.88 * grid.points**2.)
    # un-normalized 1s basis function
    model = AtomicGaussianDensity(grid.points, num_s=1, num_p=0, normalize=False)
    measure = KLDivergence()
    kl = ScipyFit(grid, dens, model, measure=measure, method="slsqp", spherical=True)
    # expected coeffs & expons
    expected_cs = np.array([2.06]) * (0.88 / np.pi)**1.5
    expected_es = np.array([0.88])
    # initial coeff=2.6 & expon=0.001
    result = kl.run(np.array([2.6]), np.array([0.001]), opt_coeffs=True, opt_expons=True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=7)
    assert_almost_equal(expected_es, result["exps"], decimal=7)
    assert_almost_equal(0., result["fun"], decimal=8)
    # initial coeff=1. & expon=0.88, opt coeffs
    result = kl.run(np.array([1.0]), np.array([0.88]), opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=8)
    assert_almost_equal(expected_es, result["exps"], decimal=8)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1.]) / (0.88 / np.pi)**1.5, result["jacobian"], decimal=6)
    # initial coeff=10. & expon=0.88, opt coeffs
    result = kl.run(np.array([10.]), np.array([0.88]), opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=8)
    assert_almost_equal(expected_es, result["exps"], decimal=8)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1.]) / (0.88 / np.pi)**1.5, result["jacobian"], decimal=6)
    # initial coeff=expected_cs & expon=expected_es, opt expons
    result = kl.run(expected_cs, expected_es, opt_coeffs=False, opt_expons=True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=8)
    assert_almost_equal(expected_es, result["exps"], decimal=8)
    assert_almost_equal(0., result["fun"], decimal=8)
    # initial coeff=expected_cs & expon=5.0, opt expons
    result = kl.run(expected_cs, np.array([5.0]), opt_coeffs=False, opt_expons=True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=8)
    assert_almost_equal(expected_es, result["exps"], decimal=8)
    assert_almost_equal(0., result["fun"], decimal=8)


def test_kl_fit_normalized_dens_normalized_1s_gaussian():
    r"""Test ScipyFiy againstnormalized Gaussian model."""
    # density is normalized 1s gaussian
    grid = UniformRadialGrid(150, 0.0, 15.0)
    dens = 2.06 * (0.88 / np.pi)**1.5 * np.exp(-0.88 * grid.points**2.)
    # normalized 1s basis function
    model = AtomicGaussianDensity(grid.points, num_s=1, num_p=0, normalize=True)
    measure = KLDivergence()
    kl = ScipyFit(grid, dens, model, measure=measure, method="slsqp", spherical=True)
    # expected coeffs & expons
    expected_cs = np.array([2.06])
    expected_es = np.array([0.88])
    # initial coeff=2.6 & expon=0.001
    result = kl.run(np.array([2.6]), np.array([0.001]), opt_coeffs=True, opt_expons=True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(expected_es, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., 0.]), result["jacobian"], decimal=6)
    # initial coeff=1. & expon=0.88, opt coeffs
    result = kl.run(np.array([1.0]), np.array([0.88]), opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(expected_es, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1.]), result["jacobian"], decimal=6)
    # initial coeff=10. & expon=0.88, opt coeffs
    result = kl.run(np.array([10.]), np.array([0.88]), opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(expected_es, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1.]), result["jacobian"], decimal=6)
    # initial coeff=expected_cs & expon=expected_es, opt expons
    result = kl.run(expected_cs, expected_es, opt_coeffs=False, opt_expons=True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(expected_es, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([0.]), result["jacobian"], decimal=6)


def test_kl_fit_normalized_dens_unnormalized_2p_gaussian():
    r"""Test ScipyFiy against un-normalized p-type Gaussian."""
    # density is normalized 2p orbitals
    grid = UniformRadialGrid(150, 1e-15, 15.0)
    points = grid.points
    cs0 = np.array([0.76, 3.09])
    es0 = np.array([2.01, 0.83])
    dens = cs0[0] * es0[0]**2.5 * points**2 * np.exp(-es0[0] * points**2)
    dens += cs0[1] * es0[1]**2.5 * points**2 * np.exp(-es0[1] * points**2)
    dens *= 2. / (3. * np.pi**1.5)
    # un-normalized 2p functions
    model = AtomicGaussianDensity(points, num_s=0, num_p=2, normalize=False)
    measure = KLDivergence()
    kl = ScipyFit(grid, dens, model, measure=measure, method="slsqp", spherical=True)
    # expected coeffs & expons
    expected_cs = cs0 * es0**2.5 * 2. / (3. * np.pi**1.5)
    expected_es = es0
    # initial coeff=[1.5, 0.1] & expon=[4., 0.001]
    result = kl.run(np.array([1.5, 0.1]), np.array([4.0, 0.001]), True, True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(expected_es, result["exps"], decimal=5)
    assert_almost_equal(0., result["fun"], decimal=8)
    # initial coeff=cs0 & expon=es0, opt coeffs
    result = kl.run(cs0, es0, opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(expected_es, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # initial coeff=[1.0, 1.0] & expon=es0, opt coeffs
    result = kl.run(np.array([1.0, 1.0]), es0, opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(expected_es, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # initial coeff=expected_cs & expon=[4.5, 1.0], opt expons
    result = kl.run(expected_cs, np.array([4.5, 1.0]), opt_coeffs=False, opt_expons=True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(expected_es, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)


def test_kl_fit_normalized_dens_normalized_2p_gaussian():
    r"""Test ScipyFiy against normalized p-type Gaussian model."""
    # density is normalized 2p orbitals
    grid = UniformRadialGrid(150, 1e-15, 15.0)

    points = grid.points
    cs0 = np.array([0.76, 3.09])
    es0 = np.array([2.01, 0.83])
    dens = cs0[0] * es0[0]**2.5 * points**2 * np.exp(-es0[0] * points**2)
    dens += cs0[1] * es0[1]**2.5 * points**2 * np.exp(-es0[1] * points**2)
    dens *= 2. / (3. * np.pi ** 1.5)
    # normalized 2p functions
    model = AtomicGaussianDensity(points, num_s=0, num_p=2, normalize=True)
    measure = KLDivergence()
    kl = ScipyFit(grid, dens, model, measure=measure, method="slsqp", spherical=True)
    # initial coeff=cs0 & expon=es0, opt coeffs
    result = kl.run(cs0, es0, True, False)
    assert_almost_equal(cs0, result["coeffs"], decimal=6)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., -1.]), result["jacobian"], decimal=6)
    # initial coeff=[2.5, 0.5] & expon=[2.0, 1.9]
    result = kl.run(np.array([2.5, 0.5]), np.array([2.0, 1.9]), True, True)
    assert_almost_equal(np.sort(cs0), np.sort(result["coeffs"]), decimal=6)
    assert_almost_equal(np.sort(es0), np.sort(result["exps"]), decimal=5)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., -1., 0., 0.]), result["jacobian"], decimal=6)
    # initial coeff=[1.0, 1.0] & expon=es0, opt coeffs
    result = kl.run(np.array([1.0, 1.0]), es0, True, False)
    assert_almost_equal(cs0, result["coeffs"], decimal=6)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., -1.]), result["jacobian"], decimal=6)
    # initial coeff=cs0 & expon=es0, opt coeffs
    result = kl.run(cs0, es0, True, False)
    assert_almost_equal(cs0, result["coeffs"], decimal=6)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., -1.]), result["jacobian"], decimal=6)


def test_kl_fit_normalized_dens_normalized_1s2p_gaussian():
    r"""Test ScipyFit with KL against normalized one s-type and 2 p-type Gaussians."""
    # density is normalized 1s + 2p gaussians
    grid = UniformRadialGrid(150, 0.0, 15.0)
    points = grid.points
    cs0 = np.array([1.52, 0.76, 3.09])
    es0 = np.array([0.50, 2.01, 0.83])
    dens = cs0[0] * (es0[0] / np.pi)**1.5 * np.exp(-es0[0] * points**2.)
    dens += cs0[1] * 2 * es0[1]**2.5 * points**2 * np.exp(-es0[1] * points**2) / (3. * np.pi**1.5)
    dens += cs0[2] * 2 * es0[2]**2.5 * points**2 * np.exp(-es0[2] * points**2) / (3. * np.pi**1.5)
    # un-normalized 1s + 2p functions
    model = AtomicGaussianDensity(points, num_s=1, num_p=2, normalize=True)
    measure = KLDivergence()
    kl = ScipyFit(grid, dens, model, measure=measure, method="slsqp", spherical=True)
    # initial coeff=1. & expon=1.
    result = kl.run(np.array([1., 1., 1.]), np.array([1., 1., 1.]), True, True)
    assert_almost_equal(np.sort(cs0), np.sort(result["coeffs"]), decimal=5)
    assert_almost_equal(np.sort(es0), np.sort(result["exps"]), decimal=5)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., -1., -1., 0., 0., 0.]), result["jacobian"], decimal=6)
    # initial coeff=[0.1, 0.6, 7.] & expon=[1., 0.9, 1.0]
    result = kl.run(np.array([0.1, 0.6, 7.]), np.array([1., 0.9, 1.0]), True, True)
    assert_almost_equal(np.sort(cs0), np.sort(result["coeffs"]), decimal=5)
    assert_almost_equal(np.sort(es0), np.sort(result["exps"]), decimal=5)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., -1., -1., 0., 0., 0.]), result["jacobian"], decimal=6)
    # initial coeff=[1., 5., 0.] & expon=es0, opt coeffs
    result = kl.run(np.array([1., 5., 0.]), es0, True, False)
    assert_almost_equal(np.sort(cs0), np.sort(result["coeffs"]), decimal=5)
    assert_almost_equal(np.sort(es0), np.sort(result["exps"]), decimal=5)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([-1., -1., -1.]), result["jacobian"], decimal=6)
    # initial coeff=cs0 & expon=es0, opt expons
    result = kl.run(cs0, es0, False, True)
    assert_almost_equal(np.sort(cs0), np.sort(result["coeffs"]), decimal=5)
    assert_almost_equal(np.sort(es0), np.sort(result["exps"]), decimal=5)
    assert_almost_equal(0., result["fun"], decimal=8)
    assert_almost_equal(np.array([0., 0., 0.]), result["jacobian"], decimal=6)


def test_kl_fit_unnormalized_1d_molecular_dens_unnormalized_1s_1s_gaussian():
    r"""Test ScipyFit of 3D un-normalized Gaussian model."""
    # density is normalized 1s + 1s gaussians
    grid = UniformRadialGrid(150, 0.0, 15.0)
    points = grid.points
    cs0 = np.array([1.52, 2.67])
    es0 = np.array([0.31, 0.41])
    coords = np.array([[0.], [1.]])
    # compute density on each center
    dens1 = cs0[0] * np.exp(-es0[0] * (points - coords[0])**2.)
    dens2 = cs0[1] * np.exp(-es0[1] * (points - coords[1])**2.)
    # un-normalized 1s + 1s functions
    model = MolecularGaussianDensity(points, coords, np.array([[1, 0], [1, 0]]), False)
    # fit total density
    measure = KLDivergence()
    kl = ScipyFit(grid, dens1 + dens2, model, measure=measure, method="slsqp", spherical=True)
    # initial coeff=1. & expon=1.
    result = kl.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(cs0, result["coeffs"], decimal=4)
    assert_almost_equal(es0, result["exps"], decimal=4)
    assert_almost_equal(0., result["fun"], decimal=8)
    # initial coeff=1. & expon=1.
    result = kl.run(np.array([5.45, 0.001]), es0, True, False, disp=True)
    assert_almost_equal(cs0, result["coeffs"], decimal=5)
    assert_almost_equal(es0, result["exps"], decimal=5)
    assert_almost_equal(0., result["fun"], decimal=8)
    # initial coeff=1. & expon=1.
    result = kl.run(cs0, np.array([5.45, 0.001]), False, True)
    assert_almost_equal(cs0, result["coeffs"], decimal=5)
    assert_almost_equal(es0, result["exps"], decimal=5)
    assert_almost_equal(0., result["fun"], decimal=8)
    # fit 1s density on center 1
    measure = KLDivergence()
    kl = ScipyFit(grid, dens1, model, measure=measure, method="slsqp", spherical=True)
    # initial coeff=1. & expon=1.
    result = kl.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(np.array([result["coeffs"][0], 0.]), result["coeffs"], decimal=4)
    assert_almost_equal(es0[0], result["exps"][0], decimal=4)
    assert_almost_equal(0., result["fun"], decimal=8)
    # initial coeff=1. & expon=1.
    result = kl.run(np.array([6.79, 8.51]), es0, True, False)
    assert_almost_equal(np.array([result["coeffs"][0], 0.]), result["coeffs"], decimal=4)
    assert_almost_equal(es0, result["exps"], decimal=4)
    assert_almost_equal(0., result["fun"], decimal=8)
    # initial coeff=1. & expon=1.
    result = kl.run(np.array([1.52, 0.0]), np.array([3.0, 4.98]), False, True)
    assert_almost_equal(np.array([result["coeffs"][0], 0.]), result["coeffs"], decimal=4)
    assert_almost_equal(es0[0], result["exps"][0], decimal=4)
    assert_almost_equal(0., result["fun"], decimal=8)


def test_kl_fit_unnormalized_1d_molecular_dens_unnormalized_1s_1p_gaussian():
    r"""Test ScipyFit against un-normalized 1S and 1P-type Gaussians."""
    # density is normalized 1s + 1s gaussians
    grid = UniformRadialGrid(150, 0.0, 15.0)
    points = grid.points
    cs0 = np.array([1.52, 2.67])
    es0 = np.array([0.31, 0.41])
    coords = np.array([[0.], [1.]])
    # compute density of each center
    dens_s = cs0[0] * (es0[0] / np.pi)**1.5 * np.exp(-es0[0] * (points - coords[0])**2.)
    dens_p = cs0[1] * (points - coords[1])**2 * np.exp(-es0[1] * (points - coords[1])**2.)
    dens_p *= (2. * es0[1]**2.5 / (3. * np.pi**1.5))
    # un-normalized 1s + 1p functions
    model = MolecularGaussianDensity(points, coords, np.array([[1, 0], [0, 1]]), True)
    # fit total density
    measure = KLDivergence()
    kl = ScipyFit(grid, dens_s + dens_p, model, measure=measure, method="slsqp", spherical=True)
    # opt. coeffs & expons
    result = kl.run(np.ones(2), np.array([0.1, 0.2]), True, True, tol=1e-20, disp=True)
    assert_almost_equal(cs0, result["coeffs"], decimal=4)
    assert_almost_equal(es0, result["exps"], decimal=4)
    assert_almost_equal(0., result["fun"], decimal=8)
    # opt. coeffs, initial expon=es0
    result = kl.run(np.array([5.91, 7.01]), es0, True, False)
    assert_almost_equal(cs0, result["coeffs"], decimal=4)
    assert_almost_equal(es0, result["exps"], decimal=4)
    assert_almost_equal(0., result["fun"], decimal=8)
    # opt. expons, initial coeff=cs0
    result = kl.run(cs0, np.array([5.91, 7.01]), False, True)
    assert_almost_equal(cs0, result["coeffs"], decimal=4)
    assert_almost_equal(es0, result["exps"], decimal=4)
    assert_almost_equal(0., result["fun"], decimal=8)
    # fit 1s density on center 1
    measure = KLDivergence()
    kl = ScipyFit(grid, dens_s, model, measure=measure, method="slsqp", spherical=True)
    # opt. coeffs & expons
    result = kl.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(np.array([cs0[0], 0.]), result["coeffs"], decimal=4)
    assert_almost_equal(es0[0], result["exps"][0], decimal=4)
    assert_almost_equal(0., result["fun"], decimal=8)
    # # fit 1p density on center 2
    measure = KLDivergence(mask_value=1e-12)
    kl = ScipyFit(grid, dens_p, model, measure=measure, method="slsqp", spherical=True)
    # opt. expons
    result = kl.run(np.array([0., cs0[1]]), np.ones(2), False, True)
    assert_almost_equal(np.array([0., cs0[1]]), result["coeffs"], decimal=4)
    assert_almost_equal(es0[1], result["exps"][1], decimal=4)
    assert_almost_equal(0., result["fun"], decimal=8)


def test_ls_fit_normalized_dens_normalized_1s_gaussian():
    r"""Test ScipyFit of least-squares against normalized Gaussian model."""
    # density is normalized 1s orbital with exponent=1.0
    grid = UniformRadialGrid(200, 0.0, 15.0)
    # actual density is a normalized 1s gaussian
    cs0, es0 = np.array([1.57]), np.array([0.51])
    dens = 1.57 * (0.51 / np.pi)**1.5 * np.exp(-0.51 * grid.points**2.)
    # model density is a normalized 1s Gaussian
    model = AtomicGaussianDensity(grid.points, num_s=1, num_p=0, normalize=True)
    measure = SquaredDifference()
    ls = ScipyFit(grid, dens, model, measure=measure, method="slsqp", spherical=True)
    # opt. coeffs & expons
    result = ls.run(np.array([0.1]), np.array([3.5]), True, True)
    assert_almost_equal(cs0, result["coeffs"], decimal=8)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # opt. coeffs
    result = ls.run(np.array([10.]), np.array([0.51]), True, False)
    assert_almost_equal(cs0, result["coeffs"], decimal=8)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # model density is two normalized 1s Gaussian
    model = AtomicGaussianDensity(grid.points, num_s=2, num_p=0, normalize=True)
    ls = ScipyFit(grid, dens, model, measure=measure, method="slsqp", spherical=True)
    # opt. coeffs & expons
    result = ls.run(np.array([0.1, 3.0]), np.array([5.1, 7.8]), True, True)
    assert_almost_equal(np.array([1.57, 0.]), result["coeffs"], decimal=6)
    assert_almost_equal(np.array([0.51, 0.]), result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)


def test_ls_fit_normalized_dens_normalized_2s_gaussian():
    r"""Test ScipyFit least-squares against two normalized s-type Gaussians."""
    # density is normalized 1s orbital with exponent=1.0
    grid = UniformRadialGrid(300, 0.0, 15.0)
    # actual density is a normalized 1s gaussian
    cs0 = np.array([1.57, 0.12])
    es0 = np.array([0.45, 1.29])
    # evaluate normalized 8s density
    dens = cs0[0] * (es0[0] / np.pi)**1.5 * np.exp(-es0[0] * grid.points**2.)
    dens += cs0[1] * (es0[1] / np.pi)**1.5 * np.exp(-es0[1] * grid.points**2.)
    # check norm of density
    assert_almost_equal(
        grid.integrate(dens * 4.0 * np.pi * grid.points**2.0), np.sum(cs0), decimal=6
    )
    # model density is a normalized 2s Gaussian
    model = AtomicGaussianDensity(grid.points, num_s=2, num_p=0, normalize=True)
    measure = SquaredDifference()
    ls = ScipyFit(grid, dens, model, measure=measure, method="slsqp", spherical=True)
    initial_cs = np.array([0.57, 0.98])
    initial_es = np.array([1.67, 0.39])
    # opt. coeffs & expons
    result = ls.run(initial_cs, initial_es, True, True)
    assert_almost_equal(np.sort(cs0), np.sort(result["coeffs"]), decimal=6)
    assert_almost_equal(np.sort(es0), np.sort(result["exps"]), decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # opt. coeffs
    result = ls.run(initial_cs, es0, True, False)
    assert_almost_equal(cs0, result["coeffs"], decimal=6)
    assert_almost_equal(es0, result["exps"], decimal=8)
    assert_almost_equal(0., result["fun"], decimal=8)


def test_ls_fit_normalized_dens_normalized_5s_gaussian():
    r"""Test ScipyFiy least-squares against 5 s-type, normalized Gaussians."""
    # density is normalized 1s orbital with exponent=1.0
    grid = UniformRadialGrid(300, 0.0, 15.0)
    # actual density is a normalized 5s gaussian
    cs0 = np.array([1.57, 0.12, 3.67, 0.97, 5.05])
    es0 = np.array([0.45, 1.29, 1.25, 20.1, 10.5])
    # evaluate normalized 5s density
    dens = cs0[0] * (es0[0] / np.pi)**1.5 * np.exp(-es0[0] * grid.points**2)
    dens += cs0[1] * (es0[1] / np.pi)**1.5 * np.exp(-es0[1] * grid.points**2)
    dens += cs0[2] * (es0[2] / np.pi)**1.5 * np.exp(-es0[2] * grid.points**2)
    dens += cs0[3] * (es0[3] / np.pi)**1.5 * np.exp(-es0[3] * grid.points**2)
    dens += cs0[4] * (es0[4] / np.pi)**1.5 * np.exp(-es0[4] * grid.points**2)
    # check norm of density
    assert_almost_equal(
        grid.integrate(dens * 4.0 * np.pi * grid.points**2.0), np.sum(cs0), decimal=6
    )
    # model density is a normalized 1s Gaussian
    model = AtomicGaussianDensity(grid.points, num_s=5, num_p=0, normalize=True)
    measure = SquaredDifference()
    ls = ScipyFit(grid, dens, model, measure=measure, method="slsqp", spherical=True)
    initial_cs = np.array([1., 0., 3., 0., 5.])
    initial_es = np.array([0.1, 1.2, 1., 20., 10.])
    # opt. coeffs
    result = ls.run(initial_cs, es0, True, False, tol=1e-15)
    assert_almost_equal(np.sort(cs0), np.sort(result["coeffs"]), decimal=6)
    assert_almost_equal(np.sort(es0), np.sort(result["exps"]), decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # opt. expons, by re-arranging the exponents and coefficients.
    initial_cs = np.array([0.12, 0.97,  3.67, 1.57, 5.05])
    initial_es = np.array([1.2, 19., 1., 0.1, 10.])
    result = ls.run(initial_cs, initial_es, False, True, tol=1e-20, with_constraint=False)
    assert_almost_equal(np.sort(initial_cs), np.sort(result["coeffs"]), decimal=6)
    assert_almost_equal(np.sort(es0), np.sort(result["exps"]), decimal=1)
    assert_almost_equal(0., result["fun"], decimal=8)


def test_ls_fit_unnormalized_1d_molecular_dens_unnormalized_2s_gaussian():
    r"""Test ScipyFit least-squares against un-normalized Gaussian model."""
    # density is normalized 1s + 1s gaussians
    grid = UniformRadialGrid(200, 0.0, 15.0)
    cs0, es0 = np.array([1.52, 2.67, ]), np.array([0.31, 0.41])
    coords = np.array([[0.], [1.]])
    # compute density on each center
    dens1 = cs0[0] * np.exp(-es0[0] * (grid.points - coords[0])**2.)
    dens2 = cs0[1] * np.exp(-es0[1] * (grid.points - coords[1])**2.)
    # un-normalized 1s + 1s basis functions
    model = MolecularGaussianDensity(grid.points, coords, np.array([[1, 0], [1, 0]]), False)
    # fit total density
    measure = SquaredDifference()
    ls = ScipyFit(grid, dens1 + dens2, model, measure=measure, method="slsqp", spherical=True)
    # initial coeff=1. & expon=1.
    result = ls.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(cs0, result["coeffs"], decimal=6)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # fit 1s density on center 1
    measure = SquaredDifference()
    ls = ScipyFit(grid, dens1, model, measure=measure, method="slsqp", spherical=True)
    # initial coeff=1. & expon=1.
    result = ls.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(np.array([1.52, 0.0]), result["coeffs"], decimal=6)
    assert_almost_equal(es0[0], result["exps"][0], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # fit 1s density on center 2
    measure = SquaredDifference()
    ls = ScipyFit(grid, dens2, model, measure=measure, method="slsqp", spherical=True)
    # initial coeff=1. & expon=1.
    result = ls.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(np.array([0.0, 2.67]), result["coeffs"], decimal=6)
    assert_almost_equal(es0[1], result["exps"][1], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)


def test_ls_fit_normalized_1d_molecular_dens_unnormalized_1s_1p_gaussian():
    r"""Test ScipyFit least-squares of 3D 1S, 1P-type Gaussian model."""
    # density is normalized 1s + 1s gaussians
    grid = UniformRadialGrid(200, 0.0, 15.0)
    cs0, es0 = np.array([1.52, 2.67]), np.array([0.31, 0.41])
    coords = np.array([[0.0], [1.0]])
    # compute density of each center
    dens_s = cs0[0] * (es0[0] / np.pi)**1.5 * np.exp(-es0[0] * (grid.points - coords[0])**2.)
    dens_p = cs0[1] * (grid.points - coords[1])**2 * np.exp(-es0[1] * (grid.points - coords[1])**2)
    dens_p *= (2. * es0[1]**2.5 / (3. * np.pi**1.5))
    # normalized 1s + 1p basis functions
    model = MolecularGaussianDensity(grid.points, coords, np.array([[1, 0], [0, 1]]), False)
    expected_cs = cs0 * np.array([(es0[0] / np.pi)**1.5, 2. * es0[1]**2.5 / (3. * np.pi**1.5)])
    # fit total density
    measure = SquaredDifference()
    ls = ScipyFit(grid, dens_s + dens_p, model, measure=measure, method="slsqp", spherical=True)
    # opt. coeffs & expons
    result = ls.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # opt. coeffs
    result = ls.run(np.array([0.001, 3.671]), es0, True, False)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # opt. expons
    result = ls.run(expected_cs, np.array([2.5, 1.9]), False, True)
    assert_almost_equal(expected_cs, result["coeffs"], decimal=6)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # fit 1s density on center 1
    measure = SquaredDifference()
    ls = ScipyFit(grid, dens_s, model, measure=measure, method="slsqp", spherical=True)
    # opt. coeffs & expons
    result = ls.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(np.array([expected_cs[0], 0.]), result["coeffs"], decimal=6)
    assert_almost_equal(es0[0], result["exps"][0], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # opt. coeffs
    result = ls.run(np.array([0.05, 5.01]), es0, True, False)
    assert_almost_equal(np.array([expected_cs[0], 0.]), result["coeffs"], decimal=6)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
    # fit 1s density on center 1
    measure = SquaredDifference()
    ls = ScipyFit(grid, dens_p, model, measure=measure, method="slsqp", spherical=True)
    result = ls.run(np.array([0.05, 5.01]), es0, True, False, with_constraint=False)
    assert_almost_equal([0., expected_cs[1]], result["coeffs"], decimal=6)
    assert_almost_equal(es0, result["exps"], decimal=6)
    assert_almost_equal(0., result["fun"], decimal=8)
