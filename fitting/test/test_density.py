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
r"""Test file for fitting.least_squares.slater_density.atomic_slater_density."""

import math
import os
import numpy as np
from fitting.density import AtomicDensity


__all__ = ["test_all_coeff_matrix_be",
           "test_atomic_density_be",
           "test_atomic_density_c",
           "test_atomic_density_ne",
           "test_phi_lcao_be",
           "test_phi_lcao_be_2",
           "test_phi_lcao_be_integrate",
           "test_phi_lcao_c_integrate",
           "test_phi_lcao_ne_integrate",
           "test_phi_matrix_be",
           "test_slater_dict_be",
           "test_slater_dict_be_2",
           "test_slater_type_orbital_be"]


# TODO ADD TESTS FOR CORE
# TODO ADD TESTS FOR VALENCE
def slater_function(exponent, n, r):
    """Calculates the normalized slater function at a given point.

    ** Arguments **

        exponent    a float or int representing the exponent of the slater function
        n           a float or int representing the natural number
        r           a float or int representing the distance of the electron from the nucleus.
    """
    assert isinstance(exponent, int) or isinstance(exponent, float)
    assert isinstance(n, int) or isinstance(n, float)
    assert isinstance(r, int) or isinstance(r, float)
    normalization = np.power(2. * exponent, n) * math.sqrt(2. * exponent / math.factorial(2*n))
    slater_orbital = np.power(r, n-1) * math.exp(-exponent * r)
    return normalization * slater_orbital


def test_slater_type_orbital_be():
    # load the Be file
    file_path = os.getcwd() + '/data/examples/be.slater'
    # using one _grid point at 1.0
    be = AtomicDensity(file_path)
    calculated = be.slator_type_orbital(np.array([[12.683501]]), np.array([[1]]), np.array([[1]]))
    expected = (2. * 12.683501)**1 * math.sqrt((2 * 12.683501) /
                                               math.factorial(2 * 1)) * 1**0 * math.exp(-12.683501)
    assert abs(calculated - expected) < 1.e-6

    be = AtomicDensity(file_path)
    calculated = be.slator_type_orbital(np.array([[0.821620]]), np.array([[2]]), np.array([[2]]))
    expected = (2. * 0.821620)**2 * math.sqrt((2 * 0.821620) /
                                              math.factorial(2 * 2)) * 2**1 * np.exp(-0.821620 * 2)
    assert abs(calculated - expected) < 1.e-6
    # using two _grid points at 1.0 and 2.0
    exp__array = np.array([[12.683501], [0.821620]])
    quantum__array = np.array([[1], [2]])
    grid = np.array([[1], [2]])
    # rows are the slator_Type orbital, where each column represents each point in the _grid

    be = AtomicDensity(file_path)
    calculated = be.slator_type_orbital(exp__array, quantum__array, grid)
    expected1 = [(2 * 12.683501)**1 * math.sqrt((2 * 12.683501) / math.factorial(2 * 1)) * 1**0 *
                 math.exp(-12.683501 * 1),
                 (2 * 12.683501)**1 * math.sqrt((2 * 12.683501) / math.factorial(2 * 1)) * 2**0 *
                 math.exp(-12.683501 * 2)]
    expected2 = [(2 * 0.821620)**2 * math.sqrt((2 * 0.821620) / math.factorial(2 * 2)) * (1**1) *
                 math.exp(-0.821620 * 1),
                 (2 * 0.821620)**2 * math.sqrt((2 * 0.821620) / math.factorial(2 * 2)) * (2**1) *
                 math.exp(-0.821620 * 2)]
    # every row corresponds to one exponent evaluated on all _grid points
    expected = np.array([expected1, expected2]).T
    assert (abs(calculated - expected) < 1.e-6).all()


def test_all_coeff_matrix_be():
    # load the Be file
    file_path = os.getcwd() + '/data/examples/be.slater'
    # using one _grid point at 1.0
    be = AtomicDensity(file_path)
    coeff_s = be.all_coeff_matrix('S')
    assert coeff_s.shape == (8, 2)
    coeff_1s = np.array([-0.0024917, 0.0314015, 0.0849694, 0.8685562, 0.0315855,
                         -0.0035284, -0.0004149, .0012299])
    coeff_2s = np.array([0.0004442, -0.0030990, -0.0367056, 0.0138910, -0.3598016,
                         -0.2563459, 0.2434108, 1.1150995])
    expected = np.hstack((coeff_1s.reshape(8, 1), coeff_2s.reshape(8, 1)))
    assert (abs(coeff_s - expected) < 1.e-6).all()


def test_phi_lcao_be():
    # load the Be file
    file_path = os.getcwd() + '/data/examples/be.slater'
    # using one _grid point at 1.0
    be = AtomicDensity(file_path)
    phi = be.phi_lcao('S', np.array([[1]]))
    assert phi.shape == (1, 2)
    r = 1
    LCAO1S = slater_function(12.683501, 1, r)*-0.0024917 + \
        slater_function(8.105927, 1, r)*0.0314015
    LCAO1S += slater_function(5.152556, 1, r)*0.0849694 + \
        slater_function(3.472467, 1, r)*0.8685562
    LCAO1S += slater_function(2.349757, 1, r)*0.0315855 + \
        slater_function(1.406429, 1, r)*-0.0035284
    LCAO1S += slater_function(0.821620, 2, r)*-0.0004149 + \
        slater_function(0.786473, 1, r)*0.0012299

    LCAO2S = slater_function(12.683501, 1, r)*0.0004442 + \
        slater_function(8.105927, 1, r)*-0.0030990
    LCAO2S += slater_function(5.152556, 1, r)*-0.0367056 + \
        slater_function(3.472467, 1, r)*0.0138910
    LCAO2S += slater_function(2.349757, 1, r)*-0.3598016 + \
        slater_function(1.406429, 1, r)*-0.2563459
    LCAO2S += slater_function(0.821620, 2, r)*0.2434108 + \
        slater_function(0.786473, 1, r)*1.1150995
    assert abs(phi[(0, 0)] - LCAO1S) < 1.e-6
    assert abs(phi[(0, 1)] - LCAO2S) < 1.e-6


def test_phi_lcao_be_2():
    # load the Be file
    file_path = os.getcwd() + '/data/examples/be.slater'
    # using three _grid points at 1.0, 2.0, and 3.0
    be = AtomicDensity(file_path)
    phi = be.phi_lcao('S', np.array([[1.], [2.], [3.]]))
    assert phi.shape == (3, 2)
    # expected values are taken from the previous example
    assert abs(phi[(0, 0)] - 0.38031668) < 1.e-6
    assert abs(phi[(0, 1)] - 0.3278693) < 1.e-6


def test_phi_matrix_be():
    # load the Be file
    file_path = os.getcwd() + '/data/examples/be.slater'
    # using one _grid point at 1.0
    be = AtomicDensity(file_path)
    # check the values of the phi_matrix
    phi_matrix = be.phi_matrix(np.array([[1]]))
    assert phi_matrix.shape == (1, 2)
    r = 1.0
    phi1S = slater_function(12.683501, 1, r) * -0.0024917 + \
        slater_function(8.105927, 1, r) * 0.0314015
    phi1S += slater_function(5.152556, 1, r) * 0.0849694 + \
        slater_function(3.472467, 1, r) * 0.8685562
    phi1S += slater_function(2.349757, 1, r) * 0.0315855 + \
        slater_function(1.406429, 1, r) * -0.0035284
    phi1S += slater_function(0.821620, 2, r) * -0.0004149 + \
        slater_function(0.786473, 1, r) * 0.00122991

    phi2S = slater_function(12.683501, 1, r) * 0.0004442 + \
        slater_function(8.105927, 1, r) * -0.0030990
    phi2S += slater_function(5.152556, 1, r) * -0.0367056 + \
        slater_function(3.472467, 1, r) * 0.0138910
    phi2S += slater_function(2.349757, 1, r) * -0.3598016 + \
        slater_function(1.406429, 1, r) * -0.2563459
    phi2S += slater_function(0.821620, 2, r) * 0.2434108 + \
        slater_function(0.786473, 1, r) * 1.1150995
    expected = np.concatenate((np.array([phi1S]), np.array([phi2S])))
    assert (abs(phi_matrix - expected) < 1.e-6).all()


def test_phi_lcao_be_integrate():
    # load the Be file
    file_path = os.getcwd() + '/data/examples/be.slater'
    # placing 100,000 equally distant points from 0.0 to 10.0 for accurate integration
    grid = np.arange(0.0, 10.0, 0.0001)
    be = AtomicDensity(file_path)
    phi = be.phi_lcao('S', grid.reshape(len(grid), 1))
    assert phi.shape == (len(grid), 2)
    # check: integrating the density of each orbital should result in one
    dens_1s = np.power(phi[:, 0], 2)
    dens_2s = np.power(phi[:, 1], 2)
    assert dens_1s.shape == dens_2s.shape == grid.shape
    # integrate_spher(r^2 * density(r)) using the composite trapezoidal rule.
    integrate_1s = np.trapz((grid**2) * dens_1s, grid)
    integrate_2s = np.trapz((grid**2) * dens_2s, grid)
    assert abs(integrate_1s - 1.0) < 1.e-4
    assert abs(integrate_2s - 1.0) < 1.e-4
    # integrate_spher(r^2 * density(r)) using the composite Simpsons rule.
    import scipy
    integrate_1s = scipy.integrate.simps((grid**2) * dens_1s, grid)
    integrate_2s = scipy.integrate.simps((grid**2) * dens_2s, grid)
    assert abs(integrate_1s - 1.0) < 1.e-4
    assert abs(integrate_2s - 1.0) < 1.e-4


def test_phi_lcao_ne_integrate():
    # load the Ne file
    file_path = os.getcwd() + '/data/examples/ne.slater'
    # placing 100,000 eqully distant points from 0.0 to 10.0 for accurate integration
    grid = np.arange(0.0, 10.0, 0.0001)
    ne = AtomicDensity(file_path)
    phi_s = ne.phi_lcao('S', grid.reshape(len(grid), 1))
    phi_p = ne.phi_lcao('P', grid.reshape(len(grid), 1))
    assert phi_s.shape == (len(grid), 2)
    assert phi_p.shape == (len(grid), 1)
    # check: integrating the density of each orbital should result in one
    dens_1s = np.power(phi_s[:, 0], 2)
    dens_2s = np.power(phi_s[:, 1], 2)
    dens_2p = np.power(np.ravel(phi_p), 2)
    assert dens_1s.shape == dens_2s.shape == grid.shape
    assert dens_2p.shape == grid.shape
    # integrate_spher(r^2 * density(r)) using the composite trapezoidal rule.
    assert abs(np.trapz(np.power(grid, 2) * dens_1s, grid) - 1.0) < 1.e-6
    assert abs(np.trapz(np.power(grid, 2) * dens_2s, grid) - 1.0) < 1.e-6
    assert abs(np.trapz(np.power(grid, 2) * dens_2p, grid) - 1.0) < 1.e-6


def test_phi_lcao_c_integrate():
    # load the C file
    file_path = os.getcwd() + '/data/examples/c.slater'
    # placing 100,000 eqully distant points from 0.0 to 10.0 for accurate integration
    grid = np.arange(0.0, 10.0, 0.0001)
    c = AtomicDensity(file_path)
    phi_s = c.phi_lcao('S', grid.reshape(len(grid), 1))
    phi_p = c.phi_lcao('P', grid.reshape(len(grid), 1))
    assert phi_s.shape == (len(grid), 2)
    assert phi_p.shape == (len(grid), 1)
    # check: integrating the density of each orbital should result in one
    dens_1s = np.power(phi_s[:, 0], 2)
    dens_2s = np.power(phi_s[:, 1], 2)
    dens_2p = np.power(np.ravel(phi_p), 2)
    assert dens_1s.shape == dens_2s.shape == grid.shape
    assert dens_2p.shape == grid.shape
    # integrate_spher(r^2 * density(r)) using the composite trapezoidal rule.
    assert abs(np.trapz(np.power(grid, 2) * dens_1s, grid) - 1.0) < 1.e-5
    assert abs(np.trapz(np.power(grid, 2) * dens_2s, grid) - 1.0) < 1.e-5
    assert abs(np.trapz(np.power(grid, 2) * dens_2p, grid) - 1.0) < 1.e-5


def test_atomic_density_be():
    # load the Be file
    file_path = os.getcwd() + '/data/examples/be.slater'
    # placing 100,000 eqully distant points from 0.0 to 10.0 for accurate integration
    grid = np.arange(0.0, 10.0, 0.0001)
    be = AtomicDensity(file_path)
    # get the density and flatten the array
    density = np.ravel(be.atomic_density(grid.reshape(len(grid), 1)))
    assert density.shape == grid.shape
    # check: integrating atomic density should result in the number of electrons.
    # integrate_spher(r^2 * density(r)) using the composite trapezoidal rule.
    assert abs(np.trapz(np.power(grid, 2) * density, grid) * 4. * np.pi - 4.0) < 1.e-3


def test_atomic_density_ne():
    # load the Ne file
    file_path = os.getcwd() + '/data/examples/ne.slater'
    # placing 100,000 eqully distant points from 0.0 to 10.0 for accurate integration
    grid = np.arange(0.0, 10.0, 0.0001)
    ne = AtomicDensity(file_path)
    # get the density and flatten the array
    density = np.ravel(ne.atomic_density(grid.reshape(len(grid), 1)))
    assert density.shape == grid.shape
    # check: integrating atomic density should result in the number of electrons.
    # integrate_spher(r^2 * density(r)) using the composite trapezoidal rule.
    assert abs(np.trapz(np.power(grid, 2) * density, grid) * 4. * np.pi - 10.0) < 1.e-6


def test_atomic_density_c():
    # load the C file
    file_path = os.getcwd() + '/data/examples/c.slater'
    # placing 100,000 eqully distant points from 0.0 to 10.0 for accurate integration
    grid = np.arange(0.0, 10.0, 0.0001)
    c = AtomicDensity(file_path)
    # get the density and flatten the array
    density = np.ravel(c.atomic_density(grid.reshape(len(grid), 1)))
    assert density.shape == grid.shape
    # check: integrating atomic density should result in the number of electrons.
    # integrate_spher(r^2 * density(r)) using the composite trapezoidal rule.
    assert abs(np.trapz(np.power(grid, 2) * density, grid) * 4. * np.pi - 6.0) < 1.e-5
