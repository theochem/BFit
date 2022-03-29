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
r"""Test bfit.density module."""

from bfit.density import SlaterAtoms
from bfit.grid import ClenshawRadialGrid
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
import scipy


def slater(e, n, r, derivative=False):
    """Calculate single normalized slater function at a given point."""
    norm = np.power(2. * e, n) * np.sqrt(2. * e / scipy.special.factorial(2 * n))
    slater = norm * np.power(r, n - 1) * np.exp(-e * r)

    if derivative:
        return (((n - 1) / r) - e) * slater
    return slater


def test_slater_type_orbital_be():
    r"""Test evaluation of Slater-type orbitals of Beryllium."""
    # load Be atomic wave function
    be = SlaterAtoms("Be")
    # check values of a single orbital at r=1.0
    orbital = be.radial_slater_orbital(np.array([[12.683501]]), np.array([[1]]), np.array([1]))
    assert_almost_equal(orbital, slater(12.683501, 1, 1.0), decimal=6)
    # check values of a single orbital at r=2.0
    orbital = be.radial_slater_orbital(np.array([[0.821620]]), np.array([[2]]), np.array([2]))
    assert_almost_equal(orbital, slater(0.821620, 2, 2.0), decimal=6)
    # check value of tow orbitals at r=1.0 & r=2.0
    exps, nums = np.array([[12.683501], [0.821620]]), np.array([[1], [2]])
    orbitals = be.radial_slater_orbital(exps, nums, np.array([1., 2.]))
    expected = np.array([[slater(exps[0, 0], nums[0, 0], 1.), slater(exps[1, 0], nums[1, 0], 1.)],
                         [slater(exps[0, 0], nums[0, 0], 2.), slater(exps[1, 0], nums[1, 0], 2.)]])
    assert_almost_equal(orbitals, expected, decimal=6)


def test_derivative_slater_type_orbital_be():
    r"""Test derivative of Slater-type orbitals of Beryllium."""
    # load Be atomic wave function
    be = SlaterAtoms("Be")
    # check values of a single orbital at r=1.0
    orbital = be.derivative_radial_slater_type_orbital(np.array([[12.683501]]),
                                                       np.array([[1]]), np.array([1]))
    assert_almost_equal(orbital, slater(12.683501, 1, 1.0, derivative=True), decimal=6)
    # check values of a single orbital at r=2.0
    orbital = be.derivative_radial_slater_type_orbital(np.array([[0.821620]]), np.array([[2]]),
                                                       np.array([2]))
    assert_almost_equal(orbital, slater(0.821620, 2, 2.0, derivative=True), decimal=6)
    # check value of single orbital at r = 0.0
    orbital = be.derivative_radial_slater_type_orbital(np.array([[0.821620]]), np.array([[2]]),
                                                       np.array([0.]))
    assert_almost_equal(orbital, 0.0, decimal=6)
    # check value of tow orbitals at r=1.0 & r=2.0
    exps, nums = np.array([[12.683501], [0.821620]]), np.array([[1], [2]])
    orbitals = be.derivative_radial_slater_type_orbital(exps, nums, np.array([1., 2.]))
    expected = np.array([[slater(exps[0, 0], nums[0, 0], 1., True),
                          slater(exps[1, 0], nums[1, 0], 1., True)],
                         [slater(exps[0, 0], nums[0, 0], 2., True),
                          slater(exps[1, 0], nums[1, 0], 2., True)]])
    assert_almost_equal(orbitals, expected, decimal=6)


def test_positive_definite_kinetic_energy_he():
    r"""Test integral of kinetic energy density of helium against actual value."""
    # load he atomic wave function
    he = SlaterAtoms("he")
    # compute density on an equally distant grid
    grid = ClenshawRadialGrid(4, 30000, 35000)
    energ = he.positive_definite_kinetic_energy(grid.points)
    integral = 4.0 * np.pi * grid.integrate(energ * grid.points**2.0)
    assert np.all(np.abs(integral - he.kinetic_energy) < 1e-5)


def test_positive_definite_kinetic_energy_li():
    r"""Test integral of kinetic energy density of lithium against actual value."""
    # load be atomic wave function
    be = SlaterAtoms("li")
    # compute density on an equally distant grid
    grid = ClenshawRadialGrid(3, 20000, 35000)
    energ = be.positive_definite_kinetic_energy(grid.points)
    integral = 4.0 * np.pi * grid.integrate(energ * grid.points**2.0)
    assert np.all(np.abs(integral - be.kinetic_energy) < 1e-5)


def test_positive_definite_kinetic_energy_c():
    r"""Test integral of kinetic energy density of carbon against actual value."""
    # load be atomic wave function
    be = SlaterAtoms("c")
    # compute density on an equally distant grid
    grid = ClenshawRadialGrid(6, 20000, 35000)
    energ = be.positive_definite_kinetic_energy(grid.points)
    integral = 4.0 * np.pi * grid.integrate(energ * grid.points**2.0)
    assert np.all(np.abs(integral - be.kinetic_energy) < 1e-5)


def test_positive_definite_kinetic_energy_p():
    r"""Test integral of kinetic energy density of phosphorous against actual value."""
    # load be atomic wave function
    be = SlaterAtoms("p")
    # compute density on ClenshawCurtis Grid
    grid = ClenshawRadialGrid(15, 20000, 35000)
    energ = be.positive_definite_kinetic_energy(grid.points)
    integral = 4.0 * np.pi * grid.integrate(energ * grid.points**2.0)
    assert np.all(np.abs(integral - be.kinetic_energy) < 1e-3)


def test_positive_definite_kinetic_energy_ag():
    r"""Test integral of kinetic energy density of silver against actual value."""
    # load c atomic wave function
    adens = SlaterAtoms("ag")
    # compute density on ClenshawCurtis Grid
    grid = ClenshawRadialGrid(47, 20000, 35000)
    energ = adens.positive_definite_kinetic_energy(grid.points)
    integral = 4.0 * np.pi * grid.integrate(energ * grid.points**2.0)
    assert np.all(np.abs(integral - adens.kinetic_energy) < 1e-3)


def test_phi_derivative_lcao_b():
    r"""Test derivative of molecular orbitals of Beryllium."""
    # load Be atomic wave function
    b = SlaterAtoms("b")
    # check the values of the phi_matrix at point 1.0
    phi_matrix = b.phi_matrix(np.array([1]), deriv=True)

    def _slater_deriv(r):
        # compute expected value of 1S
        phi1s = slater(16.109305, 2, r, True) * -0.0005529 + slater(7.628062, 1, r, True) * -0.23501
        phi1s += slater(6.135799, 2, r, True) * -0.1508924 + slater(4.167618, 1, r, True) * -0.64211
        phi1s += slater(2.488602, 1, r, True) * -0.0011507 + slater(1.642523, 2, r, True) * -0.00086
        phi1s += slater(0.991698, 1, r, True) * 0.0004712 + slater(0.787218, 1, r, True) * -0.000232
        # compute expected value of 2S
        phi2s = slater(16.109305, 2, r, True) * -0.0001239 + slater(7.628062, 1, r, True) * 0.012224
        phi2s += slater(6.135799, 2, r, True) * 0.0355967 + slater(4.167618, 1, r, True) * -0.198721
        phi2s += slater(2.488602, 1, r, True) * -0.5378967 + slater(1.642523, 2, r, True) * -0.11997
        phi2s += slater(0.991698, 1, r, True) * 1.4382402 + slater(0.787218, 1, r, True) * 0.0299258
        # compute expected value of 1P
        phi3s = slater(12.135370, 3, r, True) * 0.0000599 + slater(5.508493, 2, r, True) * 0.0113751
        phi3s += slater(3.930298, 3, r, True) * 0.0095096 + slater(2.034395, 2, r, True) * 0.1647518
        phi3s += slater(1.301082, 2, r, True) * 0.3367860 + slater(0.919434, 2, r, True) * 0.4099162
        phi3s += slater(0.787218, 2, r, True) * 0.1329396
        return [phi1s, phi2s, phi3s]
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (1, 3))
    assert_almost_equal(phi_matrix, np.array([_slater_deriv(1)]), decimal=4)
    # check the values of the phi_matrix at point 1.0, 2.0, 3.0
    phi_matrix = b.phi_matrix(np.array([1., 2., 3.]), True)
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (3, 3))
    assert_almost_equal(phi_matrix[0, :], _slater_deriv(1), decimal=4)
    assert_almost_equal(phi_matrix[1, :], _slater_deriv(2.), decimal=4)
    assert_almost_equal(phi_matrix[2, :], _slater_deriv(3.), decimal=4)


def test_coeff_matrix_be():
    r"""Test the coefficients of Beryllium is correctly parsed."""
    # load Be atomic wave function
    be = SlaterAtoms("be")
    # using one _grid point at 1.0
    coeff_1s = np.array([-0.0024917, 0.0314015, 0.0849694, 0.8685562,
                         0.0315855, -0.0035284, -0.0004149, .0012299])[:, None]
    coeff_2s = np.array([0.0004442, -0.0030990, -0.0367056, 0.0138910,
                         -0.3598016, -0.2563459, 0.2434108, 1.1150995])[:, None]
    assert_almost_equal(be.orbitals_coeff["1S"], coeff_1s, decimal=6)
    assert_almost_equal(be.orbitals_coeff["2S"], coeff_2s, decimal=6)


def test_phi_lcao_be():
    r"""Test evaluation of molecular orbitals of beryllium."""
    # load Be atomic wave function
    be = SlaterAtoms("BE")
    # check the values of the phi_matrix at point 1.0
    phi_matrix = be.phi_matrix(np.array([1]))
    # compute expected value of 1S
    phi1s = slater(12.683501, 1, 1) * -0.0024917 + slater(8.105927, 1, 1) * 0.0314015
    phi1s += slater(5.152556, 1, 1) * 0.0849694 + slater(3.472467, 1, 1) * 0.8685562
    phi1s += slater(2.349757, 1, 1) * 0.0315855 + slater(1.406429, 1, 1) * -0.0035284
    phi1s += slater(0.821620, 2, 1) * -0.0004149 + slater(0.786473, 1, 1) * 0.0012299
    # compute expected value of 2S
    phi2s = slater(12.683501, 1, 1) * 0.0004442 + slater(8.105927, 1, 1) * -0.0030990
    phi2s += slater(5.152556, 1, 1) * -0.0367056 + slater(3.472467, 1, 1) * 0.0138910
    phi2s += slater(2.349757, 1, 1) * -0.3598016 - slater(1.406429, 1, 1) * 0.2563459
    phi2s += slater(0.821620, 2, 1) * 0.2434108 + slater(0.786473, 1, 1) * 1.1150995
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (1, 2))
    assert_almost_equal(phi_matrix, np.array([[phi1s, phi2s]]), decimal=6)
    # check the values of the phi_matrix at point 1.0, 2.0, 3.0
    phi_matrix = be.phi_matrix(np.array([1., 2., 3.]))
    # check shape & orbital values
    assert_equal(phi_matrix.shape, (3, 2))
    assert_almost_equal(phi_matrix[0, :], np.array([phi1s, phi2s]), decimal=6)


def test_orbitals_norm_be():
    r"""Test that molecular orbitals are normalized of beryllium."""
    # load Be atomic wave function
    be = SlaterAtoms("be")
    # compute orbital density on an equally distant grid
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = be.phi_matrix(grid)**2
    # check shape
    assert_equal(dens.shape, (grid.size, 2))
    # check orbital normalization
    assert_almost_equal(np.trapz(grid**2 * dens[:, 0], grid), 1.0, decimal=6)
    assert_almost_equal(np.trapz(grid**2 * dens[:, 1], grid), 1.0, decimal=6)


def test_orbitals_norm_ne():
    r"""Test that molecular orbitals are normalized of neon."""
    # load Ne atomic wave function
    ne = SlaterAtoms("ne")
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
    r"""Test that molecular orbitals are normalized of carbon."""
    # load C atomic wave function
    c = SlaterAtoms("c")
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
    r"""Test integral of atomic densities matches atomic number of beryllium."""
    # load Be atomic wave function
    be = SlaterAtoms("be")
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
    r"""Test integral of atomic densities matches atomic number of neon."""
    # load Ne atomic wave function
    ne = SlaterAtoms("ne")
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
    r"""Test integral of atomic densities matches atomic number of carbon."""
    # load C atomic wave function
    c = SlaterAtoms("c")
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


def test_atomic_density_h():
    r"""Test integration of atomic density of Hydrogen."""
    h = SlaterAtoms("h")
    grid = np.arange(0.0, 15.0, 0.0001)
    dens = h.atomic_density(grid, mode="total")
    assert_almost_equal((4 * np.pi) * np.trapz(dens * grid**2., grid), 1.0, decimal=6)


def test_atomic_density_h_anion():
    r"""Test integration of atomic density of hydrogen anion."""
    assert_raises(ValueError, SlaterAtoms, "h", cation=True)  # No cations for hydrogen.
    h = SlaterAtoms("h", anion=True)
    grid = np.arange(0.0, 25.0, 0.00001)
    dens = h.atomic_density(grid, mode="total")
    assert_almost_equal((4 * np.pi) * np.trapz(dens * grid ** 2., grid), 2.0, decimal=6)


def test_atomic_density_c_anion_cation():
    r"""Test integration of atomic density of carbon anion and cation."""
    c = SlaterAtoms("c", anion=True)
    grid = np.arange(0.0, 25.0, 0.00001)
    dens = c.atomic_density(grid, mode="total")
    assert_almost_equal((4 * np.pi) * np.trapz(dens * grid ** 2., grid), 7.0, decimal=6)

    c = SlaterAtoms("c", cation=True)
    grid = np.arange(0.0, 25.0, 0.00001)
    dens = c.atomic_density(grid, mode="total")
    assert_almost_equal((4 * np.pi) * np.trapz(dens * grid ** 2., grid), 5.0, decimal=6)


def test_atomic_density_heavy_cs():
    r"""Test integration of atomic density of carbon anion and cation."""
    # These files don't exist.
    assert_raises(ValueError, SlaterAtoms, "cs", cation=True)
    assert_raises(ValueError, SlaterAtoms, "cs", anion=True)

    cs = SlaterAtoms("cs")
    grid = np.arange(0.0, 40.0, 0.0001)
    dens = cs.atomic_density(grid, mode="total")
    assert_almost_equal(4 * np.pi * np.trapz(dens * grid ** 2., grid), 55.0, decimal=5)


def test_atomic_density_heavy_rn():
    r"""Test integration of atomic density of carbon anion and cation."""
    # These files don't exist.
    assert_raises(ValueError, SlaterAtoms, "rn", cation=True)
    assert_raises(ValueError, SlaterAtoms, "rn", anion=True)

    rn = SlaterAtoms("rn")
    grid = np.arange(0.0, 40.0, 0.0001)
    dens = rn.atomic_density(grid, mode="total")
    assert_almost_equal(4 * np.pi * np.trapz(dens * grid ** 2., grid), 86, decimal=5)


def test_kinetic_energy_cation_anion_c():
    r"""Test integral of kinetic energy density of cation/anion of carbon."""
    c = SlaterAtoms("c", cation=True)
    grid = ClenshawRadialGrid(6, 20000, 35000)
    energ = c.positive_definite_kinetic_energy(grid.points)
    integral = 4.0 * np.pi * grid.integrate(energ * grid.points**2.0)
    assert_almost_equal(integral, c.kinetic_energy, decimal=6)

    c = SlaterAtoms("c", anion=True)
    grid = ClenshawRadialGrid(7, 20000, 35000)
    energ = c.positive_definite_kinetic_energy(grid.points)
    integral = 4.0 * np.pi * grid.integrate(energ * grid.points**2.0)
    assert_almost_equal(integral, c.kinetic_energy, decimal=5)


def test_derivative_electron_density_c():
    r"""Test derivative of atomic density of cation of carbon."""
    c = SlaterAtoms("c", cation=True)
    eps = 1e-10
    grid = np.array([0.1, 0.1 + eps, 0.5, 0.5 + eps])
    dens = c.atomic_density(grid)
    actual = c.derivative_density(np.array([0.1, 0.5]))
    desired_0 = (dens[1] - dens[0]) / eps
    desired_1 = (dens[3] - dens[2]) / eps
    assert_almost_equal(actual, np.array([desired_0, desired_1]), decimal=4)


def test_derivative_electron_density_cr():
    r"""Test derivative of atomic density of chromium."""
    cr = SlaterAtoms("cr")
    eps = 2.0e-8
    grid = np.array([0.1 - eps, 0.1, 0.1 + eps, 1. - eps, 1., 1. + eps])
    dens = cr.atomic_density(grid)
    actual = cr.derivative_density(np.array([0.1, 1.]))
    desired_0 = (dens[2] - dens[0]) / (2. * eps)
    desired_1 = (dens[5] - dens[3]) / (2. * eps)
    assert_almost_equal(actual, np.array([desired_0, desired_1]), decimal=4)


def test_kinetic_energy_heavy_element_ce():
    r"""Test integral of kinetic energy of cesium."""
    c = SlaterAtoms("ce")
    grid = ClenshawRadialGrid(55, 10000, 30000)
    energ = c.positive_definite_kinetic_energy(grid.points)
    integral = 4.0 * np.pi * grid.integrate(energ * grid.points**2.0)
    assert_almost_equal(integral, c.kinetic_energy, decimal=2)


def test_raises():
    r"""Test raises error of SlaterAtoms."""
    assert_raises(TypeError, SlaterAtoms, 25)
    assert_raises(TypeError, SlaterAtoms, "be2")
    assert_raises(
        ValueError, SlaterAtoms.radial_slater_orbital,
        np.array([[1]]), np.array([[2]]), np.array([[1.]])
    )
    c = SlaterAtoms("c")
    assert_raises(ValueError, c.atomic_density, np.array([[1.]]), "not total")
