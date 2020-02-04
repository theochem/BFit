# -*- coding: utf-8 -*-
# BFit is a basis-set curve-fitting optimization package.
#
# Copyright (C) 2018 The BFit Development Team.
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
r"""
Density Module.

AtomicDensity:
    Information about atoms obtained from .slater file and able to construct atomic density (total, core and valence)
        from the linear combination of Slater-type orbitals.
    Elements supported by default from "./fitting/data/examples/" range from Hydrogen to Xenon.

"""


import numpy as np
from scipy.misc import factorial

from fitting._slater import load_slater_wfn


__all__ = ["AtomicDensity"]


class AtomicDensity:
    r"""
    Atomic Density Class.

    Reads and Parses information from the .slater file of a atom and stores it inside this class.
    It is then able to construct the (total, core and valence) electron density of that element based
    on linear combination of orbitals where each orbital is a linear combination of Slater-type orbitals.
    Elements supported by default from "./fitting/data/examples/" range from Hydrogen to Xenon.

    Attributes
    ----------
    Attributes relating to the standard electron configuration.

    energy : list
        Energy of that atom.
    configuration : str
        Return the electron configuration of the element.
    orbitals : list, (M,)
        List of strings representing each of the orbitals in the electron configuration. For example, Beryllium has
        ["1S", "2S"] in its electron configuration. Ordered based on "S", "P", "D", etc.
    orbitals_occupation : ndarray, (M, 1)
        Returns the number of electrons in each of the orbitals in the electron configuration. e.g. Beryllium has two
        electrons in "1S" and two electrons in "2S".

    Attributes relating to representing orbitals as linear combination of Slater-type orbitals.

    orbital_energy : list, (N, 1)
        Energy of each of the N Slater-type orbital.
    orbitals_cusp : list, (N, 1)
        Cusp of each of the N Slater-type orbital. Same ordering as `orbitals`.
    orbitals_basis : dict
        Keys are the orbitals in the electron configuration. Each orbital has N Slater-type orbital attached to them.
    orbitals_exp : dict (str : ndarray(N, 1))
        Key is the orbital in the electron configuration and the item of that key is the Slater exponents attached
         to each N Slater-type orbital.
    orbitals_coeff : dict (str : ndarray(N, 1))
        Key is the orbital in the electron configuration (e. 1S, 2S or 2P) and the item is the Slater coefficients
        attached to each of the N Slater-type orbital.
    basis_numbers : dict (str : ndarray(N, 1))
        Key is the orbital in the electron configuration and the item is the basis number of each of the N Slater-type
        orbital. These are the principal quantum number attached to each Slater-type orbital.

    Methods
    -------
    atomic_density(mode="total") :
        Construct the atomic density from the linear combinations of slater-type orbitals. Can compute the
        total (default), core and valence atomic density.

    Examples
    --------
    Grab information about Beryllium.
    >> be =  AtomicDensity("be")

    Some of the attributes are the following.
    >> print(be.configuration) #  Should "1S(2)2S(2)".
    >> print(be.orbitals)  # ['1S', '2S'].
    >> print(be.orbitals_occupation) # [[2], [2]] Two electrons in each orbital.
    >> print(be.orbitals_cusp)  # [1.0001235, 0.9998774].

    The Slatar coefficients and exponents of the 1S orbital can be obtained as:
    >> print(be.orbital_coeff["1S"])
    >> print(be.orbitals_exp["1S"])

    The total, core and valence electron density can be obtained as:
    >> points = np.arange(0., 25., 0.01)
    >> total_density = be.atomic_density(points, "total")
    >> core_density = be.atomic_density(points, "core")
    >> valence_density = be.atomic_density(points, "valence")

    References
    ----------
    [1] "Even-Tempered Slater-Type Orbitals Revisited: From Hydrogen to Krypton" by P. Chong et al.

    [2] "Roothan-Hartree-Fock Atomic Wavefunctions. Basis functions and Their Coefficients for Ground and Certain
            Excited States of Neutral and Ionized Atoms, Z <= 54" By E. Clementi and C. Roetti.

    """

    def __init__(self, element):
        r"""
        Construct AtomicDensity object.

        Parameters
        ----------
        element : str
            Symbol of element.

        """
        if not isinstance(element, str) or not element.isalpha():
            raise TypeError("The element argument should be all letters string.")

        data = load_slater_wfn(element)
        for key, value in data.items():
            setattr(self, key, value)

    @staticmethod
    def slater_orbital(exponent, number, points):
        r"""
        Compute the Slater-type orbitals on the given points.

        A Slater-type orbital is defined as:
        .. math::
            R(r) = N r^{n-1} e^{- C r)

        where,
            :math:`n` is the principal quantum number of that orbital.
            :math:`N` is the normalizing constant.
            :math:`r` is the radial point, distance to the origin.
            :math:`C` is the zeta exponent of that orbital.

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

        See Also
        --------
        The principal quantum number of all of the orbital are stored in the attribute `basis_numbers`.
        The zeta exponents of all of the orbitals are stored in the attribute `orbitals_exp`.

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
        r"""
        Compute the linear combination of Slater-type atomic orbitals on the given points.

        Each row corresponds to a point on teh grid, represented as :math:`r` and each column is represented as a
         linear combination of Slater-type atomic orbitals of the form:

        .. math::
            \sum c_i R(r, n_i, C_i)

        where,
            :math:`c_i` is the coefficient of the Slater-type orbital.
            :math:`C_i` is the zeta exponent attached to the Slater-type orbital.
            :math:`n_i` is the principal quantum number attached to the Slater-type orbital.
            :math:`R(r, n_i, C_i)` is the Slater-type orbital.
            i ranges from 0 to K-1 where K is the number of orbitals in electron configuration.

        Parameters
        ----------
        points : ndarray, (N,)
            The radial grid points.

        Returns
        -------
        phi_matrix : ndarray(N, K)
            The linear combination of Slater-type orbitals evaluated on the grid points, where K is
            the number of orbitals.

        """
        # compute orbital composed of a linear combination of Slater
        phi_matrix = np.zeros((len(points), len(self.orbitals)))
        for index, orbital in enumerate(self.orbitals):
            exps, number = self.orbitals_exp[orbital[1]], self.basis_numbers[orbital[1]]
            slater = self.slater_orbital(exps, number, points)
            phi_matrix[:, index] = np.dot(slater, self.orbitals_coeff[orbital]).ravel()
        return phi_matrix

    def atomic_density(self, points, mode="total"):
        r"""
        Compute atomic density on the given points.

        The total density is written as a linear combination of Slater-type orbital whose coefficients is the orbital
         occupation number of the electron configuration:
        .. math::
            \sum n_i |P(r, n_i, C_i)|

        where,
            :math:`n_i` is the number of electrons in orbital i.
            :math:`P(r, n_i, C_i)` is a linear combination of Slater-type orbitals evaluated on the point :math:`r`.

        For core and valence density, please see More Info below.

        Parameters
        ----------
        points : ndarray, (N,)
            The radial grid points.
        mode : str
            The type of atomic density, which can be "total", "valence" or "core".

        Returns
        -------
        dens : ndarray, (N,)
            The atomic density on the grid points.

        Notes
        -----
        The core density and valence density is respectively written as:
        .. math::
            \sum n_i (1 - e^{-|e_i - e_{homo}|^2}) |P(r, n_i, C_i)|
            \sum n_i e^{-|e_i - e_{homo}|^2}) |P(r, n_i. C_i)|

        where,
            :math:`e_i` is the energy of the orbital i.
            :math:`e_{homo}` is the energy of the highest occupying orbital.

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
