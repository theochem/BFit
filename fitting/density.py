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
r"""Density Module.

This module computes atomic densities from Slater-type orbital basis.
"""


import numpy as np
from scipy.misc import factorial

from fitting.slater import load_slater_wfn


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

    @staticmethod
    def slater_orbital(exponent, number, points):
        r"""Compute the Slater-type orbitals on the given points.

        .. math::

        Parameters
        ----------
        exponent : ndarray, (M, 1)
            The zeta exponents of Slater orbitals.
        number : ndarray, (M, 1)
            The principle quantum numbers of Slater orbitals.
        points : ndarray, (N,)
            The radial grid points.

        Returns
        -------
        slater : ndarray, (N, M)
            The Slater-type orbitals evaluated on the grid points.
        """
        if points.ndim != 1:
            raise ValueError("The argument point should be a 1D array.")
        # compute norm & pre-factor
        norm = np.power(2. * exponent, number) * np.sqrt((2. * exponent) / factorial(2. * number))
        pref = np.power(points, number - 1).T
        # compute slater function
        slater = norm.T * pref * np.exp(-exponent * points).T
        return slater

    def phi_matrix(self, points):
        r"""Compute the linear combination of Slater-type atomic orbitals on the given points.

        .. math::

        Parameters
        ----------
        points : ndarray, (N,)
            The radial grid points.

        Returns
        -------
        phi_matrix : ndarray, (N, K)
            The linear combination of Slater-type orbitals evaluated on the grid points, where K is
            the number os orbitals.
        """
        # compute orbital composed of a linear combination of Slater
        phi_matrix = np.zeros((len(points), len(self.orbitals)))
        for index, orbital in enumerate(self.orbitals):
            exps, number = self.orbitals_exp[orbital[1]], self.basis_numbers[orbital[1]]
            slater = self.slater_orbital(exps, number, points)
            phi_matrix[:, index] = np.dot(slater, self.orbitals_coeff[orbital]).ravel()
        return phi_matrix

    def atomic_density(self, points, mode="total"):
        r"""Compute atomic density on the given points.

        .. math::

        Parameters
        ----------
        points : ndarray, (N,)
            The radial grid points.
        mode : str
            The type of atomic density, which can be "total", "valence" or "code".

        Returns
        -------
        dens : ndarray, (N,)
            The atomic density on the grid points.
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
