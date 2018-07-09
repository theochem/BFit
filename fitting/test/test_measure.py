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

from numpy.testing import assert_raises, assert_almost_equal

from fitting.measure import KLDivergence


def test_raises_kl():
    # check density argument
    assert_raises(ValueError, KLDivergence, 3.5)
    assert_raises(ValueError, KLDivergence, np.array([[1., 2., 3.]]))
    assert_raises(ValueError, KLDivergence, np.array([[1.], [2.], [3.]]))
    assert_raises(ValueError, KLDivergence, np.array([-1., 2., 3.]))
    assert_raises(ValueError, KLDivergence, np.array([1., -2., -3.]))
    # check model argument
    measure = KLDivergence(np.array([1., 2., 3.]))
    assert_raises(ValueError, measure.evaluate, 1.75)
    assert_raises(ValueError, measure.evaluate, np.array([1., 2., 3., 4.]))
    assert_raises(ValueError, measure.evaluate, np.array([[1.], [2.], [3.]]))
    assert_raises(ValueError, measure.evaluate, np.array([1., 2., -3.]))
    assert_raises(ValueError, measure.evaluate, np.array([-0.5, -1.3, -2.8]))


def test_evaluate_kl_equal():
    # test equal density & zero model
    dens = np.array([0.0, 1.0, 2.0])
    model = np.array([0., 0., 0.])
    measure = KLDivergence(dens, mask_value=1.e-12)
    assert_almost_equal(np.array([0., 0., 0.]), measure.evaluate(dens, deriv=False), decimal=8)
    assert_almost_equal(np.array([0., 0., 0.]), measure.evaluate(dens, deriv=True)[0], decimal=8)
    assert_almost_equal(-np.array([1., 1., 1.]), measure.evaluate(dens, deriv=True)[1], decimal=8)
    assert_almost_equal(np.array([0., 0., 0.]), measure.evaluate(model, deriv=False), decimal=8)
    assert_almost_equal(np.array([0., 0., 0.]), measure.evaluate(model, deriv=True)[0], decimal=8)
    assert_almost_equal(-np.array([1., 1., 1.]), measure.evaluate(model, deriv=True)[1], decimal=8)


def test_evaluate_kl():
    # test different density and model
    dens = np.linspace(0.0, 10., 15)
    model = np.linspace(0.0, 12., 15)
    # KL divergence measure
    measure = KLDivergence(dens, mask_value=1.e-12)
    k = np.array([0., -0.1302296834, -0.2604593668, -0.3906890503, -0.5209187337, -0.6511484171,
                  -0.7813781005, -0.911607784, -1.0418374674, -1.1720671508, -1.3022968342,
                  -1.4325265177, -1.5627562011, -1.6929858845, -1.8232155679])
    dk = np.array([-1.0] + [-0.8333333333] * 14)
    assert_almost_equal(k, measure.evaluate(model, deriv=False), decimal=8)
    assert_almost_equal(k, measure.evaluate(model, deriv=True)[0], decimal=8)
    assert_almost_equal(dk, measure.evaluate(model, deriv=True)[1], decimal=8)
