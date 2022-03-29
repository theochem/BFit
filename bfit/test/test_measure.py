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
r"""Test bfit.measure module."""

from bfit.measure import KLDivergence, SquaredDifference, TsallisDivergence
import numpy as np
from numpy.testing import assert_almost_equal, assert_raises


def test_raises_kl():
    r"""Test raise error when using Kullback-Leibler."""
    measure = KLDivergence()
    # check density argument
    assert_raises(ValueError, measure.evaluate, 3.5, np.ones((10,)))
    assert_raises(ValueError, measure.evaluate, np.array([[1., 2., 3.]]), np.ones((10,)))
    assert_raises(ValueError, measure.evaluate, np.array([[1.], [2.], [3.]]), np.ones((10,)))
    assert_raises(ValueError, measure.evaluate, np.array([-1., 2., 3.]), np.ones((10,)))
    assert_raises(ValueError, measure.evaluate, np.array([1., -2., -3.]), np.ones((10,)))
    # check model argument
    assert_raises(ValueError, measure.evaluate, np.array([1., 2., 3.]), 1.75)
    assert_raises(ValueError, measure.evaluate, np.array([1., 2., 3.]), np.array([1., 2., 3., 4.]))
    assert_raises(
        ValueError, measure.evaluate, np.array([1., 2., 3.]), np.array([[1.], [2.], [3.]])
    )
    # Assert that measure evaluate returns infinity when value is negative
    assert np.all(
        measure.evaluate(np.array([1., 2., 3.]), np.array([-0.5, -1.3, -2.8]))
        == measure.negative_val
    )


def test_evaluate_kl_equal():
    r"""Test evaluating Kullback-Leibler and its derivative against zero model."""
    # test equal density & zero model
    dens = np.array([0.0, 1.0, 2.0])
    model = np.array([0., 0., 0.])
    measure = KLDivergence(mask_value=1.e-12)
    assert_almost_equal(
        np.array([0., 0., 0.]), measure.evaluate(dens, dens, deriv=False), decimal=8
    )
    assert_almost_equal(
        np.array([0., 0., 0.]), measure.evaluate(dens, dens, deriv=True)[0], decimal=8
    )
    assert_almost_equal(
        -np.array([1., 1., 1.]), measure.evaluate(dens, dens, deriv=True)[1], decimal=8
    )
    assert_almost_equal(
        np.array([0., 0., 0.]), measure.evaluate(dens, model, deriv=False), decimal=8
    )
    assert_almost_equal(
        np.array([0., 0., 0.]), measure.evaluate(dens, model, deriv=True)[0], decimal=8
    )
    assert_almost_equal(
        -np.array([1., 1., 1.]), measure.evaluate(dens, model, deriv=True)[1], decimal=8
    )


def test_evaluate_kl():
    r"""Test evaluating Kullback-Leibler and its derivative against sympy results."""
    # test different density and model against sympy
    dens = np.linspace(0.0, 10., 15)
    model = np.linspace(0.0, 12., 15)
    # KL divergence measure
    measure = KLDivergence(mask_value=1.e-12)
    k = np.array([0., -0.1302296834, -0.2604593668, -0.3906890503, -0.5209187337, -0.6511484171,
                  -0.7813781005, -0.911607784, -1.0418374674, -1.1720671508, -1.3022968342,
                  -1.4325265177, -1.5627562011, -1.6929858845, -1.8232155679])
    dk = np.array([-1.0] + [-0.8333333333] * 14)
    assert_almost_equal(k, measure.evaluate(dens, model, deriv=False), decimal=8)
    assert_almost_equal(k, measure.evaluate(dens, model, deriv=True)[0], decimal=8)
    assert_almost_equal(dk, measure.evaluate(dens, model, deriv=True)[1], decimal=8)


def test_raises_squared_difference():
    r"""Test raises error in squared difference class."""
    ls = SquaredDifference()
    # check density argument
    assert_raises(ValueError, ls.evaluate, 3.5, np.ones((10,)))
    assert_raises(ValueError, ls.evaluate, np.array([[1., 2., 3.]]), np.ones((10,)))
    assert_raises(ValueError, ls.evaluate, np.array([[1.], [2.], [3.]]), np.ones((10,)))
    # check model argument
    assert_raises(ValueError, ls.evaluate, np.array([1., 2., 3.]), 1.75)
    assert_raises(ValueError, ls.evaluate, np.array([1., 2., 3.]), np.array([1., 2., 3., 4.]))
    assert_raises(ValueError, ls.evaluate, np.array([1., 2., 3.]), np.array([[1.], [2.], [3.]]))


def test_evaluate_squared_difference_equal():
    r"""Test evaluating square difference on a zero model."""
    # test equal density & zero model
    dens = np.array([0.0, 1.0, 2.0])
    model = np.array([0., 0., 0.])
    measure = SquaredDifference()
    assert_almost_equal(np.zeros(3), measure.evaluate(dens, dens, deriv=False), decimal=8)
    assert_almost_equal(np.zeros(3), measure.evaluate(dens, dens, deriv=True)[0], decimal=8)
    assert_almost_equal(np.zeros(3), measure.evaluate(dens, dens, deriv=True)[1], decimal=8)
    assert_almost_equal(dens**2, measure.evaluate(dens, model, deriv=False), decimal=8)
    assert_almost_equal(dens**2, measure.evaluate(dens, model, deriv=True)[0], decimal=8)
    assert_almost_equal(-2 * dens, measure.evaluate(dens, model, deriv=True)[1], decimal=8)


def test_evaluate_squared_difference():
    r"""Test evaluating the squared difference and its derivative against sympy."""
    # test different density and model against sympy
    dens = np.linspace(0.0, 10., 15)
    model = np.linspace(0.0, 12., 15)
    # KL divergence measure
    measure = SquaredDifference()
    # equal density
    assert_almost_equal(np.zeros(15), measure.evaluate(dens, dens, deriv=False), decimal=8)
    assert_almost_equal(np.zeros(15), measure.evaluate(dens, dens, deriv=True)[0], decimal=8)
    assert_almost_equal(np.zeros(15), measure.evaluate(dens, dens, deriv=True)[1], decimal=8)
    # different densities
    m = np.array([0.0, 0.0204081633, 0.0816326531, 0.1836734694, 0.3265306122,
                  0.5102040816, 0.7346938776, 1.0, 1.306122449, 1.6530612245,
                  2.0408163265, 2.4693877551, 2.9387755102, 3.4489795918, 4.0])
    dm = np.array([0.0, 0.2857142857, 0.5714285714, 0.8571428571, 1.1428571429,
                   1.4285714286, 1.7142857143, 2.0, 2.2857142857, 2.5714285714,
                   2.8571428571, 3.1428571429, 3.4285714286, 3.7142857143, 4.0])
    assert_almost_equal(m, measure.evaluate(dens, model, deriv=False), decimal=8)
    assert_almost_equal(m, measure.evaluate(dens, model, deriv=True)[0], decimal=8)
    assert_almost_equal(dm, measure.evaluate(dens, model, deriv=True)[1], decimal=8)


def test_raises_error_tsallis():
    r"""Test raises error when using Tsallis measure."""
    measure = TsallisDivergence()
    # check density argument
    assert_raises(ValueError, measure.evaluate, 3.5, np.ones((10,)))
    assert_raises(ValueError, measure.evaluate, np.array([[1., 2., 3.]]), np.ones((10,)))
    assert_raises(ValueError, measure.evaluate, np.array([[1.], [2.], [3.]]), np.ones((10,)))
    # check model argument
    assert_raises(ValueError, measure.evaluate, np.ones((10,)), 1.75)
    assert_raises(ValueError, measure.evaluate, np.ones((10,)), np.array([[1.], [2.], [3.]]))
    # check alpha parameter
    assert_raises(ValueError, TsallisDivergence, -10.)
    assert_raises(ValueError, TsallisDivergence, 1.0)


def test_evaluate_tsallis():
    r"""Test evaluating the Tsallis measure."""
    # Test close to one so masking plays a role.
    num_pts = 10000
    dens = np.linspace(0.0, 1., num_pts)
    # KL divergence measure
    alpha = 1.001
    measure = TsallisDivergence(alpha=alpha, mask_value=1e-8)
    # Test when model is equal to density
    assert_almost_equal(np.zeros(num_pts), measure.evaluate(dens, dens, deriv=False), decimal=8)
    # Test against brute force
    model = np.random.random(num_pts)
    # First first element to be zero, so that masking plays an important role
    model[0] = 0.
    result = []
    for i in range(num_pts):
        # Set any model values to zero if it is zero.
        if model[i] < 1e-8:
            return 0.0
        result.append(dens[i] * ((dens[i] / model[i])**(alpha - 1) - 1.0) / (alpha - 1))
    assert_almost_equal(result, measure.evaluate(dens, model, deriv=False), decimal=8)


def test_evaluating_tsallis_derivative_against_finite_difference():
    r"""Test evalauting the Tsallis measure against finite-difference."""
    num_pts = 10000
    eps = 1e-8
    dens = np.linspace(0.0, 1., num_pts)
    model = np.random.random(num_pts)
    measure = TsallisDivergence(alpha=1.001, mask_value=0.0)

    # Approximate it with central finite difference with accuracy 4.
    measure_pt_minus_2 = measure.evaluate(dens, model - 2.0 * eps, deriv=False)
    measure_pt_minus_1 = measure.evaluate(dens, model - eps, deriv=False)
    _, actual_deriv = measure.evaluate(dens, model, deriv=True)
    measure_pt_plus_1 = measure.evaluate(dens, model + eps, deriv=False)
    measure_pt_plus_2 = measure.evaluate(dens, model + 2.0 * eps, deriv=False)
    desired = (measure_pt_minus_2 / 12.0 - (2.0 / 3.0) * measure_pt_minus_1)
    desired += ((2.0 / 3.0) * measure_pt_plus_1 - measure_pt_plus_2 / 12.0)
    assert_almost_equal(actual_deriv, desired / eps, decimal=1)


def test_evaluating_tsallis_derivative_against_kullback_leibler():
    r"""Test evaluating Tsallis derivative against Kullback-Leibler."""
    num_pts = 10000
    dens = np.linspace(0.0, 1., num_pts)
    alpha = 1.000000001  # Make alpha small to approximate Kullback-Leibler
    measure = TsallisDivergence(alpha=alpha, mask_value=1e-8)
    measure_kl = KLDivergence(mask_value=1e-8)

    model = np.random.random(num_pts)
    _, deriv_tsallis = measure.evaluate(dens, model, deriv=True)
    _, deriv_kl = measure_kl.evaluate(dens, model, deriv=True)
    assert_almost_equal(deriv_tsallis, deriv_kl, decimal=1)
