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
r"""Test file for 'fitting.kl_divergence.molecular_fitting'."""


import numpy as np
import numpy.testing as npt
from fitting.grid import CubicGrid
from fitting.fit import KLDivergenceSCF
from fitting.kl_divergence.molecular_fitting import MolecularFitting


def test_input_checks():
    r"""Test for input checks for MolecularFitting."""
    true_a = CubicGrid(0., 1., 0.5)
    true_b = np.sum(true_a.points, axis=1)
    true_c = 5.
    true_d = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]])
    true_e = [5, 2, 3, 5]
    stri = "string"
    npt.assert_raises(TypeError, MolecularFitting, stri, true_b, true_c, true_d, true_e)
    npt.assert_raises(TypeError, MolecularFitting, true_a, stri, true_c, true_d, true_e)
    npt.assert_raises(TypeError, MolecularFitting, true_a, true_b, stri, true_d, true_e)
    npt.assert_raises(TypeError, MolecularFitting, true_a, true_b, true_c, stri, true_e)
    npt.assert_raises(TypeError, MolecularFitting, true_a, true_b, true_c, true_d, stri)

    bad_dimensions = np.array([[1., 3.], [2., 3.], [5., 1.], [3., 3]])
    npt.assert_raises(ValueError, MolecularFitting, true_a, true_b, true_c, bad_dimensions, true_e)

    not_two_d = np.array([5., 2.])
    npt.assert_raises(ValueError, MolecularFitting, true_a, true_b, true_c, not_two_d, true_e)

    params_not_same = [5., 2.]
    npt.assert_raises(ValueError, MolecularFitting, true_a, true_b, true_c, true_d, params_not_same)

    npt.assert_raises(ValueError, MolecularFitting, true_a, true_a.points, true_c, true_d, true_e)


def test_evaluate():
    r"""Test for getting the model for MolecularFitting."""
    grid = CubicGrid(0., 1., 0.25)
    dens_val = np.sum(grid.points, axis=1)
    norm = 1.
    mol_coord = np.array([[0., 0., 0.], [0., 0., 1.]])
    numb_of_params = [1, 2]
    dens_obj = MolecularFitting(grid, dens_val, norm, mol_coord, numb_of_params)

    c = np.array([1., 2., 3.])
    e = np.array([4., 5., 6.])
    true_answer = dens_obj.evaluate(c, e)

    radial_1 = np.sum((grid.points - mol_coord[0])**2., axis=1)
    radial_2 = np.sum((grid.points - mol_coord[1])**2., axis=1)
    c1 = c[0] * (e[0] / np.pi)**(3. / 2.)
    c2 = c[1] * (e[1] / np.pi)**(3. / 2.)
    c3 = c[2] * (e[2] / np.pi)**(3. / 2.)
    desired_answer = c1 * np.exp(-4. * radial_1) + c2 * np.exp(-5. * radial_2) + \
        c3 * np.exp(-6. * radial_2)
    npt.assert_allclose(true_answer, desired_answer)


def test_get_molecular_coordinates():
    r"""Test getting molecular coordinates for MolecularFitting."""
    grid = CubicGrid(0., 1., 0.25)
    dens_val = np.sum(grid.points, axis=1)
    norm = 1.
    mol_coord = np.array([[0., 0., 0.], [0., 0., 1.], [20., 3., 2.]])
    numb_of_params = [1, 2, 4]
    dens_obj = MolecularFitting(grid, dens_val, norm, mol_coord, numb_of_params)

    npt.assert_array_equal(dens_obj.get_mol_coord(0), mol_coord[0])
    npt.assert_array_equal(dens_obj.get_mol_coord(1), mol_coord[1])
    npt.assert_array_equal(dens_obj.get_mol_coord(2), mol_coord[1])
    npt.assert_array_equal(dens_obj.get_mol_coord(3), mol_coord[2])
    npt.assert_array_equal(dens_obj.get_mol_coord(4), mol_coord[2])
    npt.assert_array_equal(dens_obj.get_mol_coord(5), mol_coord[2])
    npt.assert_array_equal(dens_obj.get_mol_coord(6), mol_coord[2])


def test_updating_coeffs():
    r"""Test updating coefficients for MolecularFitting."""
    grid = CubicGrid(-2., 2., 0.1)
    dens_val = np.exp(-np.sum(grid.points**2., axis=1))
    norm = np.pi**(3. / 2.)
    mol_coord = np.array([[0., 0., 0.], [0., 0., 1.]])
    numb_of_params = [1, 1]
    model = MolecularFitting(grid, dens_val, norm, mol_coord, numb_of_params)
    dens_obj = KLDivergenceSCF(grid, dens_val, model)
    c = np.array([1., 2.])
    e = np.array([3., 4.])
    true_answer = dens_obj._replace_coeffs(c, e)

    radial_1 = np.sum((grid.points - mol_coord[0])**2., axis=1)
    radial_2 = np.sum((grid.points - mol_coord[1])**2., axis=1)
    promol = c[0] * (e[0] / np.pi)**1.5 * np.exp(-e[0] * radial_1) + \
        c[1] * (e[1] / np.pi)**1.5 * np.exp(-e[1] * radial_2)

    new_c = np.array([1., 2.])
    ratio = dens_val / promol
    new_c[0] *= 0.1**3. * (e[0] / np.pi)**1.5 * np.sum(ratio * np.exp(-e[0] * radial_1)) / \
        dens_obj.lagrange_multiplier
    new_c[1] *= 0.1**3. * (e[1] / np.pi)**1.5 * np.sum(ratio * np.exp(-e[1] * radial_2)) / \
        dens_obj.lagrange_multiplier
    npt.assert_array_almost_equal(true_answer, np.array([new_c, c]))


def test_updating_exponents():
    r"""Test updating exponents for MolecularFitting"""
    grid = CubicGrid(-2., 2., 0.1)
    dens_val = np.exp(-np.sum(grid.points ** 2., axis=1))
    norm = np.pi ** (3. / 2.)
    mol_coord = np.array([[0., 0., 0.], [0., 0., 1.]])
    numb_of_params = [1, 1]
    lm = grid.integrate(dens_val, spherical=True) / norm
    dens_obj = MolecularFitting(grid, dens_val, norm, mol_coord, numb_of_params)
    c = np.array([1., 2.])
    e = np.array([3., 4.])
    true_answer = dens_obj._update_fparams(c, e, lm, False)

    radial_1 = np.sum((grid.points - mol_coord[0]) ** 2., axis=1)
    radial_2 = np.sum((grid.points - mol_coord[1]) ** 2., axis=1)
    promol = c[0] * (e[0] / np.pi) ** 1.5 * np.exp(-e[0] * radial_1) + \
        c[1] * (e[1] / np.pi) ** 1.5 * np.exp(-e[1] * radial_2)

    desired_ans = np.array([0., 0.])
    ratio = dens_val / promol

    integrand = ratio * np.exp(-e[0] * radial_1)
    desired_ans[0] = 3. * np.sum(integrand) / (2. * (np.sum(integrand * radial_1)))
    integrand = ratio * np.exp(-e[1] * radial_2)
    desired_ans[1] = 3. * np.sum(integrand) / (2. * (np.sum(integrand * radial_2)))
    npt.assert_array_almost_equal(true_answer, desired_ans)
