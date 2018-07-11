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
r"""Test file for 'fitting.mbis.mbis_abc'"""


import numpy as np
import numpy.testing as npt
from fitting.model import GaussianModel
from fitting.kl_divergence.kull_leib_fitting import KullbackLeiblerFitting
from fitting.grid import BaseRadialGrid


def test_get_lagrange_multiplier():
    r"""Test the lagrange multiplier in KullbackLeiblerFitting."""
    g = BaseRadialGrid(np.arange(0., 26, 0.05))
    e = np.exp(-g.points)
    kl = KullbackLeiblerFitting(g, e, None)
    lm = 2. * 4 * np.pi / g.integrate(e, spherical=True)
    npt.assert_allclose(kl.lagrange_multiplier, lm)


def test_goodness_of_fit():
    r"""Test goodness of fit."""
    g = BaseRadialGrid(np.arange(0., 10, 0.01))
    e = np.exp(-g.points)
    m = GaussianModel(g.points, num_s=1, num_p=0)
    kl = KullbackLeiblerFitting(g, e, m)
    true_answer = kl.goodness_of_fit(np.array([1.]), np.array([1.]))[1]
    npt.assert_allclose(true_answer, 0.3431348, rtol=1e-3)


def test_goodness_of_fit_squared():
    r"""Test goodness of fit squared."""
    g = BaseRadialGrid(np.arange(0., 10, 0.01))
    e = np.exp(-g.points)
    m = GaussianModel(g.points, num_s=1, num_p=0)
    kl = KullbackLeiblerFitting(g, e, m)
    true_answer = kl.goodness_of_fit(np.array([1.]), np.array([1.]))[2]
    npt.assert_allclose(true_answer, 1.60909, rtol=1e-4)


def test_get_kullback_leibler():
    r"""Test kullback leibler formula."""
    # Test same probabiltiy distribution
    g = BaseRadialGrid(np.arange(0., 26, 0.01))
    e = np.exp(-g.points**2.)
    model = GaussianModel(g.points, num_s=1, num_p=0, normalized=False)
    kl = KullbackLeiblerFitting(g, e, model, weights=None)
    true_answer = g.integrate(kl.measure.evaluate(e, deriv=False), spherical=True)
    npt.assert_allclose(true_answer, 0.)

    # Test Different Model with wolfram
    # Integrate e^(-x^2) * log(e^(-x^2) / x) 4 pi r^2 dr from 0 to 25
    fit_model = g.points
    true_answer = g.integrate(kl.measure.evaluate(fit_model, deriv=False), spherical=True)
    npt.assert_allclose(true_answer, -0.672755 * 4 * np.pi, rtol=1e-3)


def test_get_descriptors_of_model():
    r"""Test get descriptors of model."""
    g = BaseRadialGrid(np.arange(0., 10, 0.001))
    e = np.exp(-g.points)
    model = GaussianModel(g.points, num_s=1, num_p=0, normalized=False)
    kl = KullbackLeiblerFitting(g, e, model, weights=None, mask_value=0.)
    true_answer = kl.goodness_of_fit(np.array([1.]), np.array([1.]))
    desired_answer = [5.56833, 0.3431348, 1.60909, 4. * np.pi * 17.360]
    npt.assert_allclose(true_answer, desired_answer, rtol=1e-4)


def test_run():
    g = BaseRadialGrid(np.arange(0., 10, 0.001))
    e = (1 / np.pi) ** 1.5 * np.exp(-g.points ** 2.)
    model = GaussianModel(g.points, num_s=1, num_p=0, normalized=True)
    kl = KullbackLeiblerFitting(g, e, model, weights=None)

    # Test One Basis Function
    c = np.array([1.])

    denom = np.trapz(y=g.points ** 4. * e, x=g.points)
    exps = 3. / (2. * 4. * np.pi * denom)
    params = kl.run(c, 10 * np.array([exps]), 1.e-3, 1.e-3, 1.e-8)
    params_x = params["x"]
    npt.assert_allclose(1., params_x)
    assert np.abs(params["performance"][-1, 0] - 1.) < 1e-10
    assert np.all(np.abs(params["performance"][-1][1:]) < 1e-10)
