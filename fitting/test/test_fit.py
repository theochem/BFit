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

from fitting.model import GaussianModel
from fitting.fit import KLDivergenceSCF
from fitting.grid import BaseRadialGrid


def test_lagrange_multiplier():
    g = BaseRadialGrid(np.arange(0., 26, 0.05))
    e = np.exp(-g.points)
    kl = KLDivergenceSCF(g, e, None)
    assert_almost_equal(kl.lagrange_multiplier, 1., decimal=8)
    kl = KLDivergenceSCF(g, e, None, weights=2 * np.ones_like(e))
    assert_almost_equal(kl.lagrange_multiplier, 2., decimal=8)


def test_goodness_of_fit():
    g = BaseRadialGrid(np.arange(0., 10, 0.01))
    e = np.exp(-g.points)
    m = GaussianModel(g.points, num_s=1, num_p=0, normalized=False)
    kl = KLDivergenceSCF(g, e, m, mask_value=0.)
    gf = kl.goodness_of_fit(np.array([1.]), np.array([1.]))
    expected = [5.56833, 0.3431348, 1.60909, 4. * np.pi * 17.360]
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
