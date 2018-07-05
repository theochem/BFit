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
r"""Module responsible for constructing the atomic density composed of slater functions.

This module is used to compute the atomic densities, core densities and valence densities for some
atom. Need to specify it with a .slater file.

Classes
-------
AtomicDensity : Given a file path to a slater file and grid points, Can construct an atomic,
                core, or valence densities composed of slater basis functions.

"""


import scipy
import scipy.integrate
import scipy.misc
import numpy as np
from fitting.utils.io import load_slater_wfn


__all__ = ["AtomicDensity"]


class AtomicDensity(object):
    """
    Used to compute the atomic density of some element from a composition of slater functions.

    Parameters
    ----------
    VALUES : dictionary
        Contains information about the elements:
            configuration,
            energy, orbitals,
            orbitals_energy
            orbitals_cusp,
            orbitals_basis,
            orbitals_exp,
            orbitals_coeff,
            orbitals_occupation,
            orbitals_electron_array,
            basis_numbers.

    Methods
    -------
    slater_orbital(exponent, quantumNum, r)
        Calculates the slator orbital based on its equation

    all_coeff_matrix(subshell)
        Calculates the coefficients of an atom based on its subshell.

    phi_lcao(subshell)
        Calculate molecular orbital of a specific subshell.

    phi_matrix()
        Concatenates molecular orbital horizontally.

    atomic_density()
        Calculates atomic least_squares.

    atomic_density_core()
        Calculates atomic least_squares of both core and valence.

    """
    def __init__(self, file_name):
        r"""

        Parameters
        ----------
        file_name : str
            File path to the .slater file. Primarily located in ./data/ folder.
        """
        if not isinstance(file_name, str):
            raise TypeError("File name should be a string.")

        if file_name[-2:] == "/h":
            self.electron_density = self.get_hydrogen_wave_func()
        else:
            data = load_slater_wfn(file_name)
            for key, value in data.items():
                setattr(self, key, value)

    def slater_orbital(self, exponent, number, points):
        """
        Computes the Slator Type Orbital equation.

        Parameters
        ----------
        exponent :

        quantum_num : int

        r :

        Returns
        -------
        arr
            Returns a number or an array depending on input values
        """
        normalization = (2 * exponent)**number * np.sqrt((2 * exponent) / scipy.misc.factorial(2 * number))
        pre_factor = np.transpose(points ** (np.ravel(number) - 1))
        slater = pre_factor * (np.exp(-exponent * np.transpose(points)))
        slater *= normalization
        return np.transpose(slater)

    def phi_matrix(self, points):
        """
        Connects phi equations into an array, horizontally.
        For Example, for beryllium [phi(1S), phi(2S)] is the array.
        E.G. Carbon [phi(1S), phi(2S), phi(2P)].

        Returns
        -------
        arr
             array where all of the phi equations
             for each orbital is connected together, horizontally.
             row = number of points and col = each phi equation for each orbital
        """
        # compute orbital composed of a linear combination of Slater
        phi_matrix = np.zeros((len(points), len(self.orbitals)))
        for index, orbital in enumerate(self.orbitals):
            subshell = orbital[1]
            exps, number = self.orbitals_exp[subshell], self.basis_numbers[subshell]
            slater = self.slater_orbital(exps, number, points)
            phi_matrix[:, index] = np.dot(slater, self.orbitals_coeff[orbital]).ravel()
        return phi_matrix

    def atomic_density(self, points):
        """
        By Taking the occupation numbers and multiplying it
        by the corresponding absolute, squared phi to obtain
        electron least_squares(rho).

        :return: the electron least_squares where row = number of point
                 and column = 1
        """
        dot = np.dot(self.phi_matrix(points)**2, self.orbitals_occupation)
        return np.ravel(dot) / (4. * np.pi)

    def atomic_density_core_valence(self):
        """
        Calculates Atomic Density for core and valence electrons.

        """
        def energy_homo():
            """
            A helper function that finds the HOMO energy of the element

            Returns
            --------
            float
                Energy Of Homo

            """
            # Initilize the energy from first value from the list.
            energy_homo = self.orbitals_energy['S'][0]
            for orbital, list_energy in self.orbitals_energy.items():
                max_of_list = np.max(list_energy)
                if max_of_list > energy_homo:
                    energy_homo = max_of_list
            return energy_homo

        energy_homo = energy_homo()

        def join_energy():
            """
            A helper function to join all of the energy levels into one array

            Returns
            -------
            list
                All energy levels in one list.

            """
            joined_array = np.array([])
            for orbital in ['S', 'P', 'D', 'F']:
                if orbital in self.orbitals_energy:
                    orbital_energy = self.orbitals_energy[orbital]
                    joined_array = np.hstack([joined_array, orbital_energy])
            return joined_array

        phi_matrix = self.phi_matrix()
        energy_difference = join_energy() - energy_homo

        absolute_squared = 1 - np.exp((-1) * np.absolute(energy_difference**2))
        core = absolute_squared * np.absolute(phi_matrix)**2

        absolute_squared_val = np.exp((-1) * np.absolute(energy_difference**2))
        valence = absolute_squared_val * np.absolute(phi_matrix)**2

        return(np.dot(core, self.orbitals_electron_array),
               np.dot(valence, self.orbitals_electron_array))
