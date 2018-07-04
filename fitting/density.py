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

    GRID : numpy column array
        positive _grid points used to calculate electron least_squares.

    ALL_SLATOR_ORBITALS : dictionary - key:subshell value:numpy matrix
        Calculates slator orbitals of each subshell.


    Methods
    -------
    slator_type_orbital(exponent, quantumNum, r)
        Calculates the slator orbital based on its equation.

    slator_dict()
        Calculates all slator orbital of each subshell inside a dictionary.

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
    def __init__(self, file_name, grid):
        r"""

        Parameters
        ----------
        file_name : str
            File path to the .slater file. Primarily located in ./data/ folder.

        grid : np.ndarray
            Grid Object.

        """
        if not isinstance(file_name, str):
            raise TypeError("File name should be a string.")
        if not isinstance(grid, (list, np.ndarray)):
            raise TypeError("Grid should be a list or numpy array.")
        self.row_grid = np.ravel(grid)
        self.GRID = np.reshape(np.ravel(grid), (len(grid), 1))
        if file_name[-2:] == "/h":
            self.electron_density = self.get_hydrogen_wave_func()
        else:
            self.VALUES = load_slater_wfn(file_name)
            self.ALL_SLATOR_ORBITALS = self.slator_dict()
            self.electron_density = self.atomic_density()

    def slator_type_orbital(self, exponent, quantum_num, r):
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
        normalization = ((2 * exponent) ** quantum_num) * \
            np.sqrt((2 * exponent) / scipy.misc.factorial(2 * quantum_num))
        pre_factor = np.transpose(r ** (np.ravel(quantum_num) - 1))
        slater = pre_factor * (np.exp(-exponent * np.transpose(r)))
        slater *= normalization
        return slater

    def slator_dict(self):
        """
        Groups Each Slater Equations Based On The SubShell inside an dictionary.
        This is then used to multiply by coefficient array to obtain all phi equations
        for that subshell. Hence each subshell will have their own slator matrix
        and their own coefficient matrix, dot product between them will obtain
        the phi equations(MO) for that subshell.

        :return: row = number of points, column = number of slater equations
        """
        dict_orbital = {x[1]: 0 for x in self.VALUES['orbitals']}
        for subshell in dict_orbital.keys():
            exponents = self.VALUES['orbitals_exp'][subshell]
            basis_numbers = self.VALUES['basis_numbers'][subshell]
            slater = self.slator_type_orbital(exponents, basis_numbers, self.GRID)
            dict_orbital[subshell] = np.transpose(slater)

        return dict_orbital

    def all_coeff_matrix(self, subshell):
        """
        This Groups all of the coefficients based on the subshell.
        This is then used to multiply by the specific slator array from the
        slator_dict function, in order to obtain a phi array.

        Parameters
        -----------
        subshell :
                 this is either S or P or D Or F

        Returns
        -------
        list :
            an array where row = number of coefficients per orbital and column = number of
            orbitals of specified subshell.
        """
        counter = 0
        coeffs = None

        for key in [x for x in self.VALUES['orbitals'] if x[1] == subshell]:
            if counter == 0:
                coeffs = self.VALUES['orbitals_coeff'][key]
                counter += 1
            else:
                coeffs = np.concatenate((coeffs, self.VALUES['orbitals_coeff'][key]), axis=1)
        return coeffs

    def phi_lcao(self, subshell):
        """
        Calculates phi/linear combination of atomic orbitals
        by the dot product of slator array (from slator_dict)
        and coeff array (from all_coeff_matrix(subshell)) for
        a specific subshell. Hence, to obtain all of the
        phi equations for the specific _element it must be
        repeated for each subshell (S & P & D & F).

        :return: array where row = number of points and column = number of phi/orbitals.
                For example, beryllium will have row = # of points and column = 2 (1S and 2S)
        """
        return np.dot(self.ALL_SLATOR_ORBITALS[subshell], self.all_coeff_matrix(subshell))

    def phi_matrix(self):
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
        list_orbitals = ['S', 'P', 'D', 'F']

        phi_matrix = 0
        for index, orbital in enumerate(list_orbitals):
            if orbital in self.VALUES['orbitals_exp'].keys():
                if index == 0:
                    phi_matrix = self.phi_lcao(orbital)
                else:
                    phi_matrix = np.concatenate((phi_matrix, self.phi_lcao(orbital)), axis=1)
        return phi_matrix

    def atomic_density(self):
        """
        By Taking the occupation numbers and multiplying it
        by the corresponding absolute, squared phi to obtain
        electron least_squares(rho).

        :return: the electron least_squares where row = number of point
                 and column = 1
        """
        dot = np.dot(np.absolute(self.phi_matrix())**2, self.VALUES['orbitals_electron_array'])
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
            energy_homo = self.VALUES['orbitals_energy']['S'][0]
            for orbital, list_energy in self.VALUES['orbitals_energy'].items():
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
                if orbital in self.VALUES['orbitals_energy']:
                    orbital_energy = self.VALUES['orbitals_energy'][orbital]
                    joined_array = np.hstack([joined_array, orbital_energy])
            return joined_array

        phi_matrix = self.phi_matrix()
        energy_difference = join_energy() - energy_homo

        absolute_squared = 1 - np.exp((-1) * np.absolute(energy_difference**2))
        core = absolute_squared * np.absolute(phi_matrix)**2

        absolute_squared_val = np.exp((-1) * np.absolute(energy_difference**2))
        valence = absolute_squared_val * np.absolute(phi_matrix)**2

        return(np.dot(core, self.VALUES['orbitals_electron_array']),
               np.dot(valence, self.VALUES['orbitals_electron_array']))

