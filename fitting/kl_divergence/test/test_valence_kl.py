# -*- coding: utf-8 -*-
# An basis-set curve-fitting optimization package.
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
r"""Test file for 'fitting.kl_divergence.valence_kl'."""

import numpy as np
import numpy.testing as npt
from scipy.integrate import trapz, simps
from fitting.radial_grid.general_grid import RadialGrid
from fitting.kl_divergence.valence_kl import GaussianValKL

__all__ = ["test_get_integration_factor_exps",
           "test_get_model",
           "test_get_integration_factor_coeffs",
           "test_get_norm_coeffs",
           "test_get_norm_constant",
           "test_input_valence",
           "test_updating_coefficients",
           "test_updating_exponents"]


def test_input_valence():
    r"""Test inputs for 'fitting.kl_divergence.valence_kl.GaussianValKL'."""
    g = RadialGrid(np.arange(5))
    e = g.radii**2
    i = 5.
    npt.assert_raises(TypeError, GaussianValKL, "notarray", e)
    npt.assert_raises(TypeError, GaussianValKL, g, "notarray")
    npt.assert_raises(TypeError, GaussianValKL, g, e, "notreal")
    npt.assert_raises(TypeError, GaussianValKL, g, e, i, "not")
    npt.assert_raises(ValueError, GaussianValKL, g, e, i, -5)


def test_get_norm_constant():
    r"""Test normalization constants for valence kullback-leibler."""
    g = RadialGrid(np.arange(0., 5, 0.001))
    kl = GaussianValKL(g, 3. * np.exp(-5 * g.radii**2.), 5., 1)
    # Test non-valence
    e = 5.
    true_answer = kl._get_norm_constant(e, False)

    exponent = np.exp(-e * g.radii**2.)
    ans = np.trapz(y=exponent * 4. * np.pi * g.radii**2., x=g.radii)
    npt.assert_allclose(true_answer * ans, 1, rtol=1e-5)
    npt.assert_allclose(true_answer, (e / np.pi)**(3. / 2.))

    # Test Valence
    true_answer = kl._get_norm_constant(e, True)
    ans = np.trapz(exponent * g.radii**4. * 4. * np.pi, x=g.radii)
    npt.assert_allclose(true_answer * ans, 1, rtol=1e-5)
    npt.assert_allclose(true_answer, 2 * e**(5. / 2.) / (3. * np.pi**1.5))


def test_get_norm_coeffs():
    r"""Test getting normalization constants for a list of coefficients."""
    g = RadialGrid(np.arange(0., 5, 0.001))
    kl = GaussianValKL(g, 3. * np.exp(-5 * g.radii ** 2.), 5., 1)

    c = np.array([1., 2])
    e = np.array([3., 4.])
    true_answer = kl.get_norm_coeffs(c, e)
    desired_answer = [(3. / np.pi)**1.5, 4. * 4**(5. / 2.) / (3. * np.pi**1.5)]
    npt.assert_allclose(true_answer, desired_answer)


def test_get_model():
    r"""Test getting the model for 'fitting.kl_divergence.valence_kl'."""
    g = RadialGrid(np.arange(2))
    kl = GaussianValKL(g, g.radii, inte_val=1., numb_val=1)

    c = np.array([1., 2.])
    e = np.array([3., 4.])
    true_answer = kl.get_model(c, e)
    g_s = g.radii**2.
    desired_answer = (3 / np.pi)**1.5 * np.exp(-3 * g_s) + \
                     (4 * 4**(5 / 2) / (3 * np.pi**1.5)) * g_s * np.exp(-4. * g_s)
    npt.assert_allclose(true_answer, desired_answer)


def test_get_integration_factor_coeffs():
    r"""Test coeffs integration factor for 'fitting.kl_divergence.valence_kl'."""
    g = RadialGrid(np.arange(0., 25, 0.0001))
    m = np.exp(-g.radii)
    kl = GaussianValKL(g, m, inte_val=1., numb_val=1)

    c = np.array([1., 2.])
    e = np.array([3., 4.])
    g_s = g.radii**2.
    model = c[0] * (3 / np.pi)**1.5 * np.exp(-3 * g_s) + \
        c[1] * 2. * 4**(5. / 2.) / (3. * np.pi**1.5) * g_s * np.exp(-4 * g_s)
    model[model < 1e-20] = 1e-20
    # Testing  Non-Valence Coefficients
    true_answer = kl.get_inte_factor(e[0], model, False, val=False)
    integrand = np.ma.array(m * np.exp(-3 * g_s) * g_s / model)
    desired_ans = trapz(integrand, x=g.radii)
    desired_ans *= (3. / np.pi)**1.5 * 4. * np.pi
    npt.assert_allclose(true_answer, desired_ans, rtol=1e-3)

    # Testing Valence Coefficients
    true_answer = kl.get_inte_factor(e[1], model, False, True)
    integrand = np.ma.array(m * np.exp(-4 * g_s) * g_s * g_s / model)
    desired_ans = trapz(integrand, x=g.radii)
    desired_ans *= (2 * 4.**2.5 / (3 * np.pi**1.5)) * 4. * np.pi
    npt.assert_allclose(true_answer, desired_ans, rtol=1e-3)


def test_get_integration_factor_exps():
    r"""Test exponent integration factor for 'fitting.kl_divergence.valence_kl'."""
    g = RadialGrid(np.arange(0., 25, 0.0001))
    m = np.exp(-g.radii)
    kl = GaussianValKL(g, m, inte_val=1., numb_val=1)

    c = np.array([1., 2.])
    e = np.array([3., 4.])
    g_s = g.radii ** 2.
    model = c[0] * (3 / np.pi) ** 1.5 * np.exp(-3 * g_s) + \
        c[1] * 2. * (4 ** (5. / 2.) / (3. * np.pi ** 1.5)) * g_s * np.exp(-4 * g_s)
    model[model < 1e-20] = 1e-20

    # Testing Non-Valence Exponents
    true_answer = kl.get_inte_factor(e[0], model, True, False)
    integrand = np.ma.array(m * np.exp(-3 * g_s) * g_s * g_s / model)
    desired_ans = trapz(integrand, x=g.radii)
    desired_ans *= (3. / np.pi) ** 1.5 * 4. * np.pi
    npt.assert_allclose(true_answer, desired_ans, rtol=1e-3)

    # Testing Valence Exponents
    true_answer = kl.get_inte_factor(e[1], model, True, True)
    integrand = np.ma.array(m * np.exp(-4 * g_s) * g_s * g_s * g_s / model)
    desired_ans = trapz(integrand, x=g.radii)
    desired_ans *= (2 * 4.**2.5 / (3 * np.pi**1.5)) * 4. * np.pi
    npt.assert_allclose(true_answer, desired_ans, rtol=1e-3)


def test_updating_coefficients():
    r"""Test updating coeffs for 'fitting.kl_divergence.valence_kl'."""
    g = RadialGrid(np.arange(0., 25, 0.0001))
    m = np.exp(-g.radii)
    integration = simps(m * g.radii**2. * 4. * np.pi, g.radii)
    kl = GaussianValKL(g, m, inte_val=integration, numb_val=1)
    npt.assert_allclose(kl.lagrange_multiplier, 1)

    c = np.array([1., 2.])
    e = np.array([3., 4.])
    g_s = np.ma.array(g.radii ** 2)
    model = c[0] * (3 / np.pi) ** 1.5 * np.exp(-3 * g_s) + \
        c[1] * (2. * 4**2.5 / (3. * np.pi**1.5)) * g_s * np.exp(-4 * g_s)
    npt.assert_allclose(kl.get_model(c, e), model)

    true_answer = kl._update_coeffs(c, e)[0]
    ratio = np.ma.array(m) / np.ma.array(model)
    integrand = np.ma.array(ratio * np.exp(-3 * g_s) * 4. * np.pi * g_s)
    desired_ans = simps(integrand, g.radii) * (3. / np.pi)**1.5
    c1 = c[0] * desired_ans

    integrand = np.ma.array(m * np.exp(-4 * g_s) * g_s * g_s / model)
    desired_ans = simps(integrand, x=g.radii) * (2 * 4.**2.5 / (3 * np.pi**1.5))
    c2 = c[1] * desired_ans * 4. * np.pi

    npt.assert_allclose(true_answer, [c1, c2])


def test_updating_exponents():
    r"""Test updating exponents for 'fitting.kl_divergence.valence_kl'."""
    g = RadialGrid(np.arange(0., 25, 0.0001))
    m = np.exp(-g.radii)
    integration = simps(m * g.radii ** 2. * 4. * np.pi, g.radii)
    kl = GaussianValKL(g, m, inte_val=integration, numb_val=1)
    npt.assert_allclose(kl.lagrange_multiplier, 1)

    c = np.array([1., 2.])
    e = np.array([3., 4.])
    g_s = np.ma.array(g.radii ** 2)
    model = c[0] * (3 / np.pi) ** 1.5 * np.exp(-3 * g_s) + \
        c[1] * (2. * 4 ** 2.5 / (3. * np.pi ** 1.5)) * g_s * np.exp(-4 * g_s)
    npt.assert_allclose(kl.get_model(c, e), model)

    true_answer = kl._update_exps(c, e)[0]
    ratio = np.ma.array(m) / np.ma.array(model)
    integrand = np.ma.array(ratio * np.exp(-3 * g_s) * 4. * np.pi * g_s)
    desired_answer = 3. * simps(integrand, g.radii)
    integrand2 = np.ma.array(ratio * np.exp(-3 * g_s) * 4. * np.pi * g_s * g_s)
    desired_answer /= (2. * simps(integrand2, g.radii))

    integrand2 = np.ma.array(ratio * np.exp(-4 * g_s) * g_s**2.)
    desired_answer2 = 5. * simps(integrand2, g.radii)
    integrand = np.ma.array(ratio * np.exp(-4 * g_s) * g_s * g_s * g_s)
    desired_answer2 /= (2. * simps(integrand, g.radii))

    npt.assert_allclose(true_answer, [desired_answer, desired_answer2], rtol=1e-3)
