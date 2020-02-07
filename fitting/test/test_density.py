# -*- coding: utf-8 -*-
# BFit - python program that fits a convex sum of
# positive basis functions to any probability distribution. .
#
# Copyright (C) 2020 The BFit Development Team.
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


import numpy as np

from numpy.testing import assert_equal, assert_almost_equal, assert_raises

from fitting.density import AtomicDensity


def slater(e, n, r):
    """Calculates single normalized slater function at a given point."""
    norm = np.power(2. * e, n) * np.sqrt(2. * e / np.math.factorial(2. * n))
    slater = norm * np.power(r, n - 1) * np.exp(-e * r)
    return slater


def test_slater_type_orbital_be():
    # load Be atomic wave function
    be = AtomicDensity("Be")
    # check values of a single orbital at r=1.0
    orbital = be.slater_orbital(np.array([[12.683501]]), np.array([[1]]), np.array([1]))
    assert_almost_equal(orbital, slater(12.683501, 1, 1.0), decimal=6)
    # check values of a single orbital at r=2.0
    orbital = be.slater_orbital(np.array([[0.821620]]), np.array([[2]]), np.array([2]))
    assert_almost_equal(orbital, slater(0.821620, 2, 2.0), decimal=6)
    # check value of tow orbitals at r=1.0 & r=2.0
    exps, nums = np.array([[12.683501], [0.821620]]), np.array([[1], [2]])
    orbitals = be.slater_orbital(exps, nums, np.array([1., 2.]))
    expected = np.array([[slater(exps[0, 0], nums[0, 0], 1.), slater(exps[1, 0], nums[1, 0], 1.)],
                         [slater(exps[0, 0], nums[0, 0], 2.), slater(exps[1, 0], nums[1, 0], 2.)]])
    assert_almost_equal(orbitals, expected, decimal=6)


def test_coeff_matrix_be():
    # load Be atomic wave function
    be = AtomicDensity("be")
    # using one _grid point at 1.0
    coeff_1s = np.array([-0.0024917, 0.0314015, 0.0849694, 0.8685562,
                         0.0315855, -0.0035284, -0.0004149, .0012299])[:, None]
    coeff_2s = np.array([0.0004442, -0.0030990, -0.0367056, 0.0138910,
                         -0.3598016, -0.2563459, 0.2434108, 1.1150995])[:, None]
    assert_almost_equal(be.orbitals_coeff["1S"], coeff_1s, decimal=6)
    assert_almost_equal(be.orbitals_coeff["2S"], coeff_2s, decimal=6)


def test_phi_lcao_be():
    # load Be atomic wave function
    be = AtomicDensity("BE")
    # check the values of the phi_matrix at point 1.0
    phi_matrix = be.phi_matrix(np.array([1]))
    # compute expected value of 1S
    phi1S = slater(12.683501, 1, 1) * -0.0024917 + slater(8.105927, 1, 1) * 0.0314015
    phi1S += slater(5.152556, 1, 1) * 0.0849694 + slater(3.472467, 1, 1) * 0.8685562
    phi1S += slater(2.349757, 1, 1) * 0.0315855 + slater(1.406429, 1, 1) * -0.0035284
    phi1S += slater(0.821620, 2, 1) * -0.0004149 + slater(0.786473, 1, 1) * 0.0012299
    # compute expected value of 2S
    phi2S = slater(12.683501, 1, 1) * 0.0004442 + slater(8.105927, 1, 1) * -0.0030990
    phi2S += slater(5.152556, 1, 1) * -0.0367056 + slater(3.472467, 1, 1) * 0.0138910
    phi2S += slater(2.349757, 1, 1) * -0.3598016 - slater(1.406429, 1, 1) * 0.2563459
    phi2S += slater(0.821620, 2, 1) * 0.2434108 + slater(0.786473, 1, 1) * 1.1150995
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (1, 2))
    assert_almost_equal(phi_matrix, np.array([[phi1S, phi2S]]), decimal=6)
    # check the values of the phi_matrix at point 1.0, 2.0, 3.0
    phi_matrix = be.phi_matrix(np.array([1., 2., 3.]))
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (3, 2))
    assert_almost_equal(phi_matrix[0, :], np.array([phi1S, phi2S]), decimal=6)


def test_orbitals_function_be():
    # load Be atomic wave function
    be = AtomicDensity("bE")
    # check the values of the phi_matrix at point 1.0
    phi_matrix = be.phi_matrix(np.array([1]))
    # compute expected value of 1S
    phi1S = slater(12.683501, 1, 1.) * -0.0024917 + slater(8.105927, 1, 1.) * 0.0314015
    phi1S += slater(5.152556, 1, 1.) * 0.0849694 + slater(3.472467, 1, 1.) * 0.8685562
    phi1S += slater(2.349757, 1, 1.) * 0.0315855 + slater(1.406429, 1, 1.) * -0.0035284
    phi1S += slater(0.821620, 2, 1.) * -0.0004149 + slater(0.786473, 1, 1.) * 0.00122991
    # compute expected value of 2S
    phi2S = slater(12.683501, 1, 1.) * 0.0004442 + slater(8.105927, 1, 1.) * -0.0030990
    phi2S += slater(5.152556, 1, 1.) * -0.0367056 + slater(3.472467, 1, 1.) * 0.0138910
    phi2S += slater(2.349757, 1, 1.) * -0.3598016 + slater(1.406429, 1, 1.) * -0.2563459
    phi2S += slater(0.821620, 2, 1.) * 0.2434108 + slater(0.786473, 1, 1.) * 1.1150995
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (1, 2))
    assert_almost_equal(phi_matrix, np.array([[phi1S, phi2S]]), decimal=6)


def test_orbitals_norm_be():
    # load Be atomic wave function
    be = AtomicDensity("be")
    # compute orbital density on an equally distant grid
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = be.phi_matrix(grid)**2
    # check shape
    assert_equal(dens.shape, (grid.size, 2))
    # check orbital normalization
    assert_almost_equal(np.trapz(grid**2 * dens[:, 0], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 1], grid), 1.0, decimal=6)


def test_orbitals_norm_ne():
    # load Ne atomic wave function
    ne = AtomicDensity("ne")
    # compute orbital density on an equally distant grid
    grid = np.arange(0.0, 10.0, 0.0001)
    dens = ne.phi_matrix(grid)**2
    # check shape
    assert_equal(dens.shape, (grid.size, 3))
    # check orbital normalization
    assert_almost_equal(np.trapz(grid**2 * dens[:, 0], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 1], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 2], grid), 1.0, decimal=6)


def test_orbitals_norm_c():
    # load C atomic wave function
    c = AtomicDensity("c")
    # compute orbital density on an equally distant grid
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = c.phi_matrix(grid)**2
    # check shape
    assert_equal(dens.shape, (grid.size, 3))
    # check orbital normalization
    assert_almost_equal(np.trapz(grid**2 * dens[:, 0], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 1], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 2], grid), 1.0, decimal=6)


def test_atomic_density_be():
    # load Be atomic wave function
    be = AtomicDensity("be")
    # compute density on an equally distant grid
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = be.atomic_density(grid, mode="total")
    core = be.atomic_density(grid, mode="core")
    valn = be.atomic_density(grid, mode="valence")
    # check shape
    assert_equal(dens.shape, grid.shape)
    assert_equal(core.shape, grid.shape)
    assert_equal(valn.shape, grid.shape)
    # check dens = core + valence
    assert_almost_equal(dens, core + valn, decimal=6)
    # check number of electrons
    assert_almost_equal(4 * np.pi * np.trapz(grid**2 * dens, grid), 4.0, decimal=6)


def test_atomic_density_ne():
    # load Ne atomic wave function
    ne = AtomicDensity("ne")
    # compute density on an equally distant grid
    grid = np.arange(0.0, 10.0, 0.0001)
    dens = ne.atomic_density(grid, mode="total")
    core = ne.atomic_density(grid, mode="core")
    valn = ne.atomic_density(grid, mode="valence")
    # check shape
    assert_equal(dens.shape, grid.shape)
    assert_equal(core.shape, grid.shape)
    assert_equal(valn.shape, grid.shape)
    # check dens = core + valence
    assert_almost_equal(dens, core + valn, decimal=6)
    # check number of electrons
    assert_almost_equal(4 * np.pi * np.trapz(grid**2 * dens, grid), 10.0, decimal=6)


def test_atomic_density_c():
    # load C atomic wave function
    c = AtomicDensity("c")
    # compute density on an equally distant grid
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = c.atomic_density(grid, mode="total")
    core = c.atomic_density(grid, mode="core")
    valn = c.atomic_density(grid, mode="valence")
    # check shape
    assert_equal(dens.shape, grid.shape)
    assert_equal(core.shape, grid.shape)
    assert_equal(valn.shape, grid.shape)
    # check dens = core + valence
    assert_almost_equal(dens, core + valn, decimal=6)
    # check number of electrons
    assert_almost_equal(4 * np.pi * np.trapz(grid**2 * dens, grid), 6.0, decimal=6)


def test_raises():
    assert_raises(TypeError, AtomicDensity, 25)
    assert_raises(TypeError, AtomicDensity, "be2")
    assert_raises(ValueError, AtomicDensity.slater_orbital, np.array([[1]]), np.array([[2]]), np.array([[1.]]))
    c = AtomicDensity("c")
    assert_raises(ValueError, c.atomic_density, np.array([[1.]]), "not total")
