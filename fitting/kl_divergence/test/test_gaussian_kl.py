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
r"""Test file for 'fitting.divergence_fitting.gaussian_kl"""

import numpy as np
import numpy.testing as npt
from scipy.integrate import simps, quad
from fitting.grid import BaseRadialGrid
from fitting.model import GaussianModel
from fitting.kl_divergence.kull_leib_fitting import KullbackLeiblerFitting


def test_get_model():
    coeff = np.array([5., 2., 3., 50.])
    expon = np.array([10., 3., 2., 1.])
    g = BaseRadialGrid(np.arange(0., 10.))
    kl = GaussianModel(g.points, num_s=4, num_p=0, normalized=True)
    true_answer = kl.evaluate(coeff, expon)
    normalized_coeffs = np.array([coeff[0] * (expon[0] / np.pi) ** (3. / 2.),
                                  coeff[1] * (expon[1] / np.pi) ** (3. / 2.),
                                  coeff[2] * (expon[2] / np.pi) ** (3. / 2.),
                                  coeff[3] * (expon[3] / np.pi) ** (3. / 2.)])
    exponential = np.exp(-expon * g.points.reshape((len(g.points), 1)) ** 2.)
    desired_answer = exponential.dot(normalized_coeffs)
    npt.assert_array_almost_equal(true_answer, desired_answer)


def test_get_integration_factor_coeffs():
    r"""Test getting the integration factor for coefficients."""
    c = np.array([5., 2.])
    e = np.array([10., 3.])
    g = BaseRadialGrid(np.arange(0., 25, 1e-4))
    e2 = np.exp(-g.points)
    m = GaussianModel(g.points, num_s=2, num_p=0, normalized=True)
    kl = KullbackLeiblerFitting(g, e2, m)

    # Integration Factor for updating coefficient.
    model = c[0] * (e[0] / np.pi) ** (3. / 2.) * np.exp(-e[0] * g.points ** 2.) + \
        c[1] * (e[1] / np.pi) ** (3. / 2.) * np.exp(-e[1] * g.points ** 2.)
    true_answer = kl.get_inte_factor(e[0], model, False)
    true_answer2 = kl.get_inte_factor(e[1], model, False)

    # Testing with Simps and Masked Array
    masked_arr = np.ma.array(e2 * np.exp(-e[0] * g.points ** 2.))
    desired_answer = simps(y=masked_arr * g.points ** 2. / model, x=g.points)
    desired_answer *= 4. * np.pi * (e[0] / np.pi) ** (3. / 2.)
    npt.assert_allclose(true_answer, desired_answer, rtol=1e-5)

    masked_arr = np.ma.array(e2 * np.exp(-e[1] * g.points ** 2.))
    desired_answer = simps(y=masked_arr * g.points ** 2. / model, x=g.points)
    desired_answer *= 4. * np.pi * (e[1] / np.pi) ** (3. / 2.)
    npt.assert_allclose(true_answer2, desired_answer, rtol=1e-5)

    # Testing with Simps and Zero Division
    model[model == 0] = 1e-20
    desired_answer = simps(y=e2 * np.exp(-e[0] * g.points ** 2.) * g.points ** 2. / model, x=g.points)
    desired_answer *= 4. * np.pi * (e[0] / np.pi) ** (3. / 2.)
    npt.assert_allclose(true_answer, desired_answer, rtol=1e-3)

    # Test with lambda function
    def f(x, ex):
        m = c[0] * (e[0] / np.pi) ** (3. / 2.) * np.exp(-e[0] * x ** 2.) + \
            c[1] * (e[1] / np.pi) ** (3. / 2.) * np.exp(-e[1] * x ** 2.)
        em = np.exp(-x) * 4. * np.pi * x ** 2. * (ex / np.pi) ** (3. / 2.)
        em *= np.exp(-ex * x ** 2.)
        return em / m

    desired_answer = quad(f, 0, 5, args=(e[0]))
    npt.assert_allclose(true_answer, desired_answer[0], rtol=1e-5)
    desired_answer = quad(f, 0, 15, args=(e[1]))
    npt.assert_allclose(true_answer2, desired_answer[0], rtol=1e-3)


def test_get_integration_factor_exps():
    r"""Test getting integration factor for updating exponents."""
    coeff = np.array([5., 3.])
    exps = np.array([10., 8])
    points = np.arange(0., 5, 1e-3)
    grid = BaseRadialGrid(points)
    tmod = np.exp(-points)
    m = GaussianModel(grid.points, num_s=2, num_p=0, normalized=True)
    kl = KullbackLeiblerFitting(grid, tmod, m)

    model = coeff[0] * ((exps[0] / np.pi) ** (3. / 2.)) * np.exp(-exps[0] * points ** 2.) + \
        coeff[1] * ((exps[1] / np.pi) ** (3. / 2.)) * np.exp(-exps[1] * points ** 2.)
    true_answer = kl.get_inte_factor(exps[0], model, True)
    true_answer2 = kl.get_inte_factor(exps[1], model, True)

    # Test with using simpson and masked array
    integrand = tmod * np.exp(-exps[0] * points ** 2.) * points ** 4. / model
    desired_answer1 = simps(integrand, points)
    desired_answer1 *= 4. * np.pi * (exps[0] / np.pi) ** (3. / 2.)
    npt.assert_allclose(true_answer, desired_answer1)

    integrand = tmod * np.exp(-exps[1] * points ** 2.) * points ** 4. / model
    desired_answer2 = simps(integrand, points)
    desired_answer2 *= 4. * np.pi * (exps[1] / np.pi) ** (3. / 2.)
    npt.assert_allclose(true_answer2, desired_answer2)

    # Test using quad rule
    def f(x, ex):
        m = coeff[0] * (exps[0] / np.pi) ** (3. / 2.) * np.exp(-exps[0] * x ** 2.) + \
            coeff[1] * (exps[1] / np.pi) ** (3. / 2.) * np.exp(-exps[1] * x ** 2.)

        em = np.exp(-x) * 4. * np.pi * x ** 4. * (ex / np.pi) ** (3. / 2.)
        em *= np.exp(-ex * x ** 2.)
        return em / m

    desired_answer1 = quad(f, 0, 5, args=(exps[0]))
    npt.assert_allclose(true_answer, desired_answer1[0], rtol=1e-4)
    desired_answer2 = quad(f, 0, 5, args=(exps[1]))
    npt.assert_allclose(true_answer2, desired_answer2[0], rtol=1e-2)


def test_update_coeff():
    c = np.array([5., 2.])
    e = np.array([10., 3.])
    g = BaseRadialGrid(np.arange(0., 9, 0.001))
    e2 = np.exp(-g.points)
    m = GaussianModel(g.points, num_s=2, num_p=0, normalized=True)
    kl = KullbackLeiblerFitting(g, e2, m)

    model = c[0] * (e[0] / np.pi) ** (3. / 2.) * np.exp(-e[0] * g.points ** 2.) + \
            c[1] * (e[1] / np.pi) ** (3. / 2.) * np.exp(-e[1] * g.points ** 2.)
    true_answer = kl._update_coeffs(c, e)

    desired_ans = c.copy()
    integrand = e2 * np.exp(-e[0] * g.points ** 2.) * g.points ** 2. / model
    desired_answer1 = simps(integrand, g.points)
    desired_answer1 *= 4. * np.pi * (e[0] / np.pi) ** (3. / 2.)

    desired_ans[0] *= desired_answer1

    integrand = e2 * np.exp(-e[1] * g.points ** 2.) * g.points ** 2. / model
    desired_answer2 = simps(integrand, g.points)
    desired_answer2 *= 4. * np.pi * (e[1] / np.pi) ** (3. / 2.)
    desired_ans[1] *= desired_answer2
    npt.assert_allclose(desired_ans, true_answer, rtol=1e-3)


def test_update_func_params():
    c = np.array([5., 2.])
    e = np.array([10., 3.])
    g = BaseRadialGrid(np.arange(0., 13, 0.001))
    e2 = np.exp(-g.points)
    m = GaussianModel(g.points, num_s=2, num_p=0, normalized=True)
    kl = KullbackLeiblerFitting(g, e2, m)

    model = c[0] * (e[0] / np.pi) ** (3. / 2.) * np.exp(-e[0] * g.points ** 2.) + \
        c[1] * (e[1] / np.pi) ** (3. / 2.) * np.exp(-e[1] * g.points ** 2.)
    model = np.ma.array(model)
    # Assume without convergence
    true_answer = kl._update_fparams(c, e, False)

    # Find Numerator of integration factor
    integrand = e2 * np.exp(-e[0] * g.points ** 2.) * g.points ** 2. / model
    desired_answer_num = simps(integrand, g.points)
    desired_answer_num *= 4. * np.pi * (e[0] / np.pi) ** (3. / 2.)
    # Find Denomenator of integrate factor
    integrand = e2 * np.exp(-e[0] * g.points ** 2.) * g.points ** 4. / model
    desired_answer_den1 = simps(integrand, g.points)
    desired_answer_den1 *= 4. * np.pi * (e[0] / np.pi) ** (3. / 2.)
    # Update first Exponent
    desired_answer1 = 3. * desired_answer_num / (2. * desired_answer_den1)

    # Find Numerator of integration factor.
    integrand = e2 * np.exp(-e[1] * g.points ** 2.) * g.points ** 2. / model
    desired_answer_num = simps(integrand, g.points)
    desired_answer_num *= 4. * np.pi * (e[1] / np.pi) ** (3. / 2.)
    # Find Denomenator of integration factor.
    integrand = e2 * np.exp(-e[1] * g.points ** 2.) * g.points ** 4. / model
    desired_answer_den = simps(integrand, g.points)
    desired_answer_den *= 4. * np.pi * (e[1] / np.pi) ** (3. / 2.)
    # Update Second Exponent
    desired_answer2 = 3. * desired_answer_num / (2. * desired_answer_den)

    npt.assert_allclose(true_answer, [desired_answer1, desired_answer2])

    # Assume With Convergence
    true_answer = kl._update_fparams(c, e, True)
    desired_answer1 = 3. / (2. * desired_answer_den1)
    desired_answer2 = 3. / (2. * desired_answer_den)
    npt.assert_allclose(true_answer, [desired_answer1, desired_answer2], rtol=1e-2)
