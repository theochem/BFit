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
"""


import numpy as np

from scipy.misc import factorial

from fitting.utils.io import load_slater_wfn


__all__ = ["AtomicDensity"]


class AtomicDensity(object):
    """Atomic Density Class."""

    def __init__(self, element):
        r"""
        Parameters
        ----------
        element : str
            Symbol of element.
        """
        if not isinstance(element, str) or not element.isalpha():
            raise TypeError("The element argument should be all letters string.")

        if element.lower() == "h":
            raise NotImplementedError
        else:
            data = load_slater_wfn(element)
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
        # compute norm & prefactor
        norm = np.power(2. * exponent, number) * np.sqrt((2. * exponent) / factorial(2. * number))
        pref = np.power(points, number - 1).T
        # compute slater function
        slater = norm.T * pref * np.exp(-exponent * points).T
        return slater

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
            exps, number = self.orbitals_exp[orbital[1]], self.basis_numbers[orbital[1]]
            slater = self.slater_orbital(exps, number, points)
            phi_matrix[:, index] = np.dot(slater, self.orbitals_coeff[orbital]).ravel()
        return phi_matrix

    def atomic_density(self, points, mode="total"):
        """Compute atomic density on the given 1D grid points array.

        Parameters
        ----------
        points
        """
        if mode not in ["total", "valence", "core"]:
            raise ValueError("Argument mode not recognized!")

        # compute orbital occupation numbers
        orb_occs = self.orbitals_occupation
        if mode == "valence":
            orb_homo = self.orbitals_energy[len(self.orbitals_occupation) - 1]
            orb_occs = orb_occs * np.exp(-(self.orbitals_energy - orb_homo)**2)
        elif mode == "core":
            orb_homo = self.orbitals_energy[len(self.orbitals_occupation) - 1]
            orb_occs = orb_occs * (1. - np.exp(-(self.orbitals_energy - orb_homo)**2))
        # compute density
        dens = np.dot(self.phi_matrix(points)**2, orb_occs).ravel() / (4 * np.pi)
        return dens
