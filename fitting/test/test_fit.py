# -*- coding: utf-8 -*-
# FittingBasisSets is a basis-set curve-fitting optimization package.
#
# Copyright (C) 2018 The FittingBasisSets Development Team.
#
# This file is part of FittingBasisSets.
#
# FittingBasisSets is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# FittingBasisSets is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# ---


import numpy as np
from numpy.testing import assert_almost_equal

from fitting.model import GaussianModel, MolecularGaussianModel
from fitting.fit import KLDivergenceSCF, KLDivergenceFit
from fitting.grid import BaseRadialGrid


def test_lagrange_multiplier():
    g = BaseRadialGrid(np.arange(0., 26, 0.05))
    e = np.exp(-g.points)
    kl = KLDivergenceSCF(g, e, None)
    assert_almost_equal(kl.lagrange_multiplier, 1., decimal=8)
    kl = KLDivergenceSCF(g, e, None, weights=2 * np.ones_like(e))
    assert_almost_equal(kl.lagrange_multiplier, 2., decimal=8)


def test_goodness_of_fit():
    g = BaseRadialGrid(np.arange(0., 10, 0.01), spherical=True)
    e = np.exp(-g.points)
    m = GaussianModel(g.points, num_s=1, num_p=0, normalized=False)
    kl = KLDivergenceSCF(g, e, m, mask_value=0.)
    gf = kl.goodness_of_fit(np.array([1.]), np.array([1.]))
    expected = [5.56833, 4 * np.pi * 1.60909, 4. * np.pi * 17.360]
    assert_almost_equal(expected, gf, decimal=1)


def test_run_normalized_1s_gaussian():
    # density is normalized 1s orbital with exponent=1.0
    g = BaseRadialGrid(np.arange(0., 10, 0.001))
    e = (1. / np.pi)**1.5 * np.exp(-g.points**2.)
    model = GaussianModel(g.points, num_s=1, num_p=0, normalized=True)
    kl = KLDivergenceSCF(g, e, model, weights=None)

    # fit density with initial coeff=1. & expon=1.
    res = kl.run(np.array([1.]), np.array([1.]), 1.e-4, 1.e-4, 1.e-4)
    # check optimized coeffs & expons
    assert_almost_equal(np.array([1.]), res["x"][0], decimal=8)
    assert_almost_equal(np.array([1.]), res["x"][1], decimal=8)
    # check value of optimized objective function & fitness measure
    assert_almost_equal(0., res["fun"][-1], decimal=10)
    assert_almost_equal(1., res["performance"][-1, 0], decimal=8)
    assert_almost_equal(0., res["performance"][-1, 1:], decimal=8)

    # fit density with initial coeff=0.5 & expon=0.5
    res = kl.run(np.array([0.5]), np.array([0.5]), 1.e-4, 1.e-4, 1.e-4)
    # check optimized coeffs & expons
    assert_almost_equal(np.array([1.]), res["x"][0], decimal=8)
    assert_almost_equal(np.array([1.]), res["x"][1], decimal=8)
    # check value of optimized objective function & fitness measure
    assert_almost_equal(0., res["fun"][-1], decimal=10)
    assert_almost_equal(1., res["performance"][-1, 0], decimal=8)
    assert_almost_equal(0., res["performance"][-1, 1:], decimal=8)

    # fit density with initial coeff=0.1 & expon=10.
    res = kl.run(np.array([0.1]), np.array([10.]), 1.e-4, 1.e-4, 1.e-4)
    # check optimized coeffs & expons
    assert_almost_equal(np.array([1.]), res["x"][0], decimal=8)
    assert_almost_equal(np.array([1.]), res["x"][1], decimal=8)
    # check value of optimized objective function
    assert_almost_equal(0., res["fun"][-1], decimal=10)

    # fit density with initial coeff=20. & expon=0.01.
    res = kl.run(np.array([20.]), np.array([0.01]), 1.e-4, 1.e-4, 1.e-4)
    # check optimized coeffs & expons
    assert_almost_equal(np.array([1.]), res["x"][0], decimal=8)
    assert_almost_equal(np.array([1.]), res["x"][1], decimal=8)
    # check value of optimized objective function
    assert_almost_equal(0., res["fun"][-1], decimal=10)


def test_kl_fit_unnormalized_dens_normalized_1s_gaussian():
    # density is normalized 1s orbital with exponent=1.0
    grid = BaseRadialGrid(np.arange(0., 10, 0.001), spherical=True)
    # density is normalized 1s gaussian
    dens = 1.57 * np.exp(-0.51 * grid.points**2.)
    model = GaussianModel(grid.points, num_s=1, num_p=0, normalized=True)
    kl = KLDivergenceFit(grid, dens, model, "slsqp")
    # expected coeffs & expons
    expected_cs = np.array([1.57 / (0.51 / np.pi)**1.5])
    expected_es = np.array([0.51])
    # initial coeff=1.57 & expon=0.51
    cs, es, f, df = kl.run(np.array([1.57]), np.array([0.51]), True, True)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., 0.]), df, decimal=6)
    # initial coeff=0.1 & expon=0.1
    cs, es, f, df = kl.run(np.array([0.1]), np.array([0.1]), True, True)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., 0.]), df, decimal=6)
    # initial coeff=5.0 & expon=15.
    cs, es, f, df = kl.run(np.array([5.0]), np.array([15.]), True, True)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., 0.]), df, decimal=6)
    # initial coeff=0.8 & expon=0.51, opt coeffs
    cs, es, f, df = kl.run(np.array([0.5]), np.array([0.51]), True, False)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1.]), df, decimal=6)
    # # initial coeff=1.57 & expon=1., opt expons
    # cs, es, f, df = kl.run(np.array([1.57]), np.array([1.]), False, True)
    # assert_almost_equal(expected_cs, cs, decimal=8)
    # assert_almost_equal(expected_es, es, decimal=8)
    # assert_almost_equal(0., f, decimal=10)
    # assert_almost_equal(np.array([-1.]), df, decimal=6)


def test_kl_fit_normalized_dens_unnormalized_1s_gaussian():
    # density is normalized 1s gaussian
    grid = BaseRadialGrid(np.arange(0., 10, 0.001), spherical=True)
    dens = 2.06 * (0.88 / np.pi)**1.5 * np.exp(-0.88 * grid.points**2.)
    # un-normalized 1s basis function
    model = GaussianModel(grid.points, num_s=1, num_p=0, normalized=False)
    kl = KLDivergenceFit(grid, dens, model, "slsqp")
    # expected coeffs & expons
    expected_cs = np.array([2.06]) * (0.88 / np.pi)**1.5
    expected_es = np.array([0.88])
    # initial coeff=2.6 & expon=0.001
    cs, es, f, df = kl.run(np.array([2.6]), np.array([0.001]), opt_coeffs=True, opt_expons=True)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    # initial coeff=1. & expon=0.88, opt coeffs
    cs, es, f, df = kl.run(np.array([1.0]), np.array([0.88]), opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1.]) / (0.88 / np.pi)**1.5, df, decimal=6)
    # initial coeff=10. & expon=0.88, opt coeffs
    cs, es, f, df = kl.run(np.array([10.]), np.array([0.88]), opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1.]) / (0.88 / np.pi)**1.5, df, decimal=6)
    # initial coeff=expected_cs & expon=expected_es, opt expons
    cs, es, f, df = kl.run(expected_cs, expected_es, opt_coeffs=False, opt_expons=True)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    # initial coeff=expected_cs & expon=5.0, opt expons
    cs, es, f, df = kl.run(expected_cs, np.array([5.0]), opt_coeffs=False, opt_expons=True)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)


def test_kl_fit_normalized_dens_normalized_1s_gaussian():
    # density is normalized 1s gaussian
    grid = BaseRadialGrid(np.arange(0., 10, 0.001), spherical=True)
    dens = 2.06 * (0.88 / np.pi)**1.5 * np.exp(-0.88 * grid.points**2.)
    # normalized 1s basis function
    model = GaussianModel(grid.points, num_s=1, num_p=0, normalized=True)
    kl = KLDivergenceFit(grid, dens, model, "slsqp")
    # expected coeffs & expons
    expected_cs = np.array([2.06])
    expected_es = np.array([0.88])
    # initial coeff=2.6 & expon=0.001
    cs, es, f, df = kl.run(np.array([2.6]), np.array([0.001]), opt_coeffs=True, opt_expons=True)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., 0.]), df, decimal=6)
    # initial coeff=1. & expon=0.88, opt coeffs
    cs, es, f, df = kl.run(np.array([1.0]), np.array([0.88]), opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1.]), df, decimal=6)
    # initial coeff=10. & expon=0.88, opt coeffs
    cs, es, f, df = kl.run(np.array([10.]), np.array([0.88]), opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1.]), df, decimal=6)
    # initial coeff=expected_cs & expon=expected_es, opt expons
    cs, es, f, df = kl.run(expected_cs, expected_es, opt_coeffs=False, opt_expons=True)
    assert_almost_equal(expected_cs, cs, decimal=8)
    assert_almost_equal(expected_es, es, decimal=8)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([0.]), df, decimal=6)
    # initial coeff=expected_cs & expon=5.0, opt expons
    # cs, es, f, df = kl.run(expected_cs, np.array([5.0]), opt_coeffs=False, opt_expons=True)
    # assert_almost_equal(expected_cs, cs, decimal=8)
    # assert_almost_equal(expected_es, es, decimal=8)
    # assert_almost_equal(0., f, decimal=10)
    # assert_almost_equal(np.array([-1.]), df, decimal=6)


def test_kl_fit_normalized_dens_unnormalized_2p_gaussian():
    # density is normalized 2p orbitals
    grid = BaseRadialGrid(np.arange(0., 10, 0.001), spherical=True)
    points = grid.points
    cs0 = np.array([0.76, 3.09])
    es0 = np.array([2.01, 0.83])
    dens = cs0[0] * es0[0]**2.5 * points**2 * np.exp(-es0[0] * points**2)
    dens += cs0[1] * es0[1]**2.5 * points**2 * np.exp(-es0[1] * points**2)
    dens *= 2. / (3. * np.pi**1.5)
    # un-normalized 2p functions
    model = GaussianModel(points, num_s=0, num_p=2, normalized=False)
    kl = KLDivergenceFit(grid, dens, model, "slsqp")
    # expected coeffs & expons
    expected_cs = cs0 * es0**2.5 * 2. / (3. * np.pi**1.5)
    expected_es = es0
    # initial coeff=[1.5, 0.1] & expon=[4., 0.001]
    cs, es, f, df = kl.run(np.array([1.5, 0.1]), np.array([4.0, 0.001]), True, True)
    assert_almost_equal(expected_cs, cs, decimal=6)
    assert_almost_equal(expected_es, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    # initial coeff=cs0 & expon=es0, opt coeffs
    cs, es, f, df = kl.run(cs0, es0, opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, cs, decimal=6)
    assert_almost_equal(expected_es, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    # initial coeff=[1.0, 1.0] & expon=es0, opt coeffs
    cs, es, f, df = kl.run(np.array([1.0, 1.0]), es0, opt_coeffs=True, opt_expons=False)
    assert_almost_equal(expected_cs, cs, decimal=6)
    assert_almost_equal(expected_es, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    # initial coeff=expected_cs & expon=[4.5, 1.0], opt expons
    cs, es, f, df = kl.run(expected_cs, np.array([0., 0.]), opt_coeffs=False, opt_expons=True)
    assert_almost_equal(expected_cs, cs, decimal=6)
    assert_almost_equal(expected_es, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)


def test_kl_fit_normalized_dens_normalized_2p_gaussian():
    # density is normalized 2p orbitals
    grid = BaseRadialGrid(np.arange(0., 10, 0.001), spherical=True)
    points = grid.points
    cs0 = np.array([0.76, 3.09])
    es0 = np.array([2.01, 0.83])
    dens = cs0[0] * es0[0]**2.5 * points**2 * np.exp(-es0[0] * points**2)
    dens += cs0[1] * es0[1]**2.5 * points**2 * np.exp(-es0[1] * points**2)
    dens *= 2. / (3. * np.pi ** 1.5)
    # normalized 2p functions
    model = GaussianModel(points, num_s=0, num_p=2, normalized=True)
    kl = KLDivergenceFit(grid, dens, model, "slsqp")
    # initial coeff=cs0 & expon=es0, opt coeffs
    cs, es, f, df = kl.run(cs0, es0, True, False)
    assert_almost_equal(cs0, cs, decimal=6)
    assert_almost_equal(es0, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., -1.]), df, decimal=6)
    # initial coeff=[2.5, 0.5] & expon=[2.0, 1.9]
    cs, es, f, df = kl.run(np.array([2.5, 0.5]), np.array([2.0, 1.9]), True, True)
    assert_almost_equal(np.sort(cs0), np.sort(cs), decimal=6)
    assert_almost_equal(np.sort(es0), np.sort(es), decimal=5)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., -1., 0., 0.]), df, decimal=6)
    # initial coeff=[1.0, 1.0] & expon=es0, opt coeffs
    cs, es, f, df = kl.run(np.array([1.0, 1.0]), es0, True, False)
    assert_almost_equal(cs0, cs, decimal=6)
    assert_almost_equal(es0, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., -1.]), df, decimal=6)
    # initial coeff=cs0 & expon=es0, opt coeffs
    cs, es, f, df = kl.run(cs0, es0, True, False)
    assert_almost_equal(cs0, cs, decimal=6)
    assert_almost_equal(es0, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., -1.]), df, decimal=6)
    # initial coeff=cs0 & expon=[2.0, 0.8]
    # cs, es, f, df = kl.run(cs0, np.array([2.0, 0.8]), False, True)


def test_kl_fit_normalized_dens_normalized_1s2p_gaussian():
    # density is normalized 1s + 2p gaussians
    grid = BaseRadialGrid(np.arange(0., 10, 0.001), spherical=True)
    points = grid.points
    cs0 = np.array([1.52, 0.76, 3.09])
    es0 = np.array([0.50, 2.01, 0.83])
    dens = cs0[0] * (es0[0] / np.pi)**1.5 * np.exp(-es0[0] * points**2.)
    dens += cs0[1] * 2 * es0[1]**2.5 * points**2 * np.exp(-es0[1] * points**2) / (3. * np.pi**1.5)
    dens += cs0[2] * 2 * es0[2]**2.5 * points**2 * np.exp(-es0[2] * points**2) / (3. * np.pi**1.5)
    # un-normalized 1s + 2p functions
    model = GaussianModel(points, num_s=1, num_p=2, normalized=True)
    kl = KLDivergenceFit(grid, dens, model, "slsqp")
    # initial coeff=1. & expon=1.
    cs, es, f, df = kl.run(np.array([1., 1., 1.]), np.array([1., 1., 1.]), True, True)
    assert_almost_equal(cs0, cs, decimal=6)
    assert_almost_equal(es0, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., -1., -1., 0., 0., 0.]), df, decimal=6)
    # initial coeff=[0.1, 0.6, 7.] & expon=[1., 0.9, 1.0]
    cs, es, f, df = kl.run(np.array([0.1, 0.6, 7.]), np.array([1., 0.9, 1.0]), True, True)
    assert_almost_equal(np.sort(cs0), np.sort(cs), decimal=5)
    assert_almost_equal(np.sort(es0), np.sort(es), decimal=5)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., -1., -1., 0., 0., 0.]), df, decimal=6)
    # initial coeff=[1., 5., 0.] & expon=es0, opt coeffs
    cs, es, f, df = kl.run(np.array([1., 5., 0.]), es0, True, False)
    assert_almost_equal(np.sort(cs0), np.sort(cs), decimal=5)
    assert_almost_equal(np.sort(es0), np.sort(es), decimal=5)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([-1., -1., -1.]), df, decimal=6)
    # initial coeff=cs0 & expon=es0, opt expons
    cs, es, f, df = kl.run(cs0, es0, False, True)
    assert_almost_equal(np.sort(cs0), np.sort(cs), decimal=5)
    assert_almost_equal(np.sort(es0), np.sort(es), decimal=5)
    assert_almost_equal(0., f, decimal=10)
    assert_almost_equal(np.array([0., 0., 0.]), df, decimal=6)
    # # initial coeff=cs0 & expon=es0, opt expons
    # cs, es, f, df = kl.run(cs0, np.ones(3), False, True)
    # assert_almost_equal(np.array([cs0[1], cs0[0], cs0[2]]), cs, decimal=6)
    # assert_almost_equal(np.array([es0[1], es0[0], es0[2]]), es, decimal=6)
    # assert_almost_equal(0., f, decimal=10)
    # assert_almost_equal(np.array([0., 0., 0.]), df, decimal=6)


def test_kl_fit_unnormalized_1d_molecular_dens_unnormalized_1s_1s_gaussian():
    # density is normalized 1s + 1s gaussians
    grid = BaseRadialGrid(np.arange(0., 10, 0.001), spherical=True)
    points = grid.points
    cs0 = np.array([1.52, 2.67])
    es0 = np.array([0.31, 0.41])
    coords = np.array([0., 1.])
    # compute density on each center
    dens1 = cs0[0] * np.exp(-es0[0] * (points - coords[0])**2.)
    dens2 = cs0[1] * np.exp(-es0[1] * (points - coords[1])**2.)
    # un-normalized 1s + 1s functions
    model = MolecularGaussianModel(points, coords, np.array([[1, 0], [1, 0]]), False)
    # fit total density
    kl = KLDivergenceFit(grid, dens1 + dens2, model, "slsqp")
    # initial coeff=1. & expon=1.
    cs, es, f, df = kl.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(cs0, cs, decimal=4)
    assert_almost_equal(es0, es, decimal=4)
    assert_almost_equal(0., f, decimal=10)
    # initial coeff=1. & expon=1.
    cs, es, f, df = kl.run(np.array([5.45, 0.001]), es0, True, False)
    assert_almost_equal(cs0, cs, decimal=6)
    assert_almost_equal(es0, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    # initial coeff=1. & expon=1.
    cs, es, f, df = kl.run(cs0, np.array([5.45, 0.001]), False, True)
    assert_almost_equal(cs0, cs, decimal=6)
    assert_almost_equal(es0, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    # fit 1s density on center 1
    kl = KLDivergenceFit(grid, dens1, model, "slsqp")
    # initial coeff=1. & expon=1.
    cs, es, f, df = kl.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(np.array([cs[0], 0.]), cs, decimal=4)
    assert_almost_equal(es0[0], es[0], decimal=4)
    assert_almost_equal(0., f, decimal=10)
    # initial coeff=1. & expon=1.
    cs, es, f, df = kl.run(np.array([6.79, 8.51]), es0, True, False)
    assert_almost_equal(np.array([cs[0], 0.]), cs, decimal=4)
    assert_almost_equal(es0, es, decimal=4)
    assert_almost_equal(0., f, decimal=10)
    # initial coeff=1. & expon=1.
    cs, es, f, df = kl.run(np.array([1.52, 0.0]), np.array([3.0, 4.98]), False, True)
    assert_almost_equal(np.array([cs[0], 0.]), cs, decimal=4)
    assert_almost_equal(es0[0], es[0], decimal=4)
    assert_almost_equal(0., f, decimal=10)


def test_kl_fit_unnormalized_1d_molecular_dens_unnormalized_1s_1p_gaussian():
    # density is normalized 1s + 1s gaussians
    grid = BaseRadialGrid(np.arange(0., 10, 0.001), spherical=True)
    points = grid.points
    cs0 = np.array([1.52, 2.67])
    es0 = np.array([0.31, 0.41])
    coords = np.array([0., 1.])
    # compute density of each center
    dens_s = cs0[0] * (es0[0] / np.pi)**1.5 * np.exp(-es0[0] * (points - coords[0])**2.)
    dens_p = cs0[1] * (points - coords[1])**2 * np.exp(-es0[1] * (points - coords[1])**2.)
    dens_p *= (2. * es0[1]**2.5 / (3. * np.pi**1.5))
    # un-normalized 1s + 1p functions
    model = MolecularGaussianModel(points, coords, np.array([[1, 0], [0, 1]]), True)
    # fit total density
    kl = KLDivergenceFit(grid, dens_s + dens_p, model, "slsqp")
    # opt. coeffs & expons
    cs, es, f, df = kl.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(cs0, cs, decimal=6)
    assert_almost_equal(es0, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    # opt. coeffs, initial expon=es0
    cs, es, f, df = kl.run(np.array([5.91, 7.01]), es0, True, False)
    assert_almost_equal(cs0, cs, decimal=6)
    assert_almost_equal(es0, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    # opt. expons, initial coeff=cs0
    cs, es, f, df = kl.run(cs0, np.array([5.91, 7.01]), False, True)
    assert_almost_equal(cs0, cs, decimal=6)
    assert_almost_equal(es0, es, decimal=6)
    assert_almost_equal(0., f, decimal=10)
    # fit 1s density on center 1
    kl = KLDivergenceFit(grid, dens_s, model, "slsqp")
    # opt. coeffs & expons
    cs, es, f, df = kl.run(np.ones(2), np.ones(2), True, True)
    assert_almost_equal(np.array([cs0[0], 0.]), cs, decimal=6)
    assert_almost_equal(es0[0], es[0], decimal=6)
    assert_almost_equal(0., f, decimal=10)
    # # fit 1p density on center 2
    kl = KLDivergenceFit(grid, dens_p, model, "slsqp", mask_value=1.e-12)
    # opt. expons
    cs, es, f, df = kl.run(np.array([0., cs0[1]]), np.ones(2), False, True)
    assert_almost_equal(np.array([0., cs0[1]]), cs, decimal=6)
    assert_almost_equal(es0[1], es[1], decimal=6)
    assert_almost_equal(0., f, decimal=10)
