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
r"""Slater Atomic Density Module."""

from bfit._slater import load_slater_wfn
import numpy as np
from scipy.special import factorial


__all__ = ["SlaterAtoms"]


class SlaterAtoms:
    r"""
    Atomic Density Class.

    Reads and parses information from the .slater file [1], [2] of an atom and stores it inside
    this class. Elements supported by default from "./bfit/data/examples/" range from Hydrogen
    to Xenon.

    Each electron of the atom is associated to a molecular spin-orbital written as:

    .. math::
        \phi_i(r, \theta, \phi) = \bigg[\sum_{j=1}^{M_i} c^i_j R_{n^i_j}(r, \alpha^i_j) \bigg]
        Y_{l_i}^{m_i}(\theta, \phi) \sigma(m_i),

    where :math:`R_{n^i_j}(r)` is a Slater-type orbital with quantum number :math:`n^i_j`,
    :math:`c^i_j` is the coefficient of the jth Slater-type orbital,
    :math:`\alpha_j^i` is the exponent of the jth Slater-type orbital,
    :math:`Y_{l_i}^{m_i}` is the spherical harmonic with complex form, angular momentum :math:`l_i`
    determined by the electron in the electron configuration and :math:`m_i` is the spin determined
    by applying Hund's rule to the electron configuration, and
    :math:`\sigma` is the spin-function determined from applying Hund's rule to the electron.

    References
    ----------
    .. [1] Koga, T. , Kanayama, K. , Watanabe, S. and Thakkar, A. J. (1999),
       Analytical Hartree–Fock wave functions subject to cusp and asymptotic
       constraints: He to Xe, Li+ to Cs+, H−  Int. J. Quantum Chem., 71: 491-497.
       doi:10.1002/(SICI)1097-461X(1999)71:6<491::AID-QUA6>3.0.CO;2-T

    .. [2] Koga, T., Kanayama, K., Watanabe, T. et al. Analytical Hartree–Fock wave functions
       for the atoms Cs to Lr. Theor Chem Acc 104, 411–413 (2000).
       https://doi.org/10.1007/s002140000150

    """

    def __init__(self, element, anion=False, cation=False):
        r"""
        Construct SlaterAtoms object.

        Parameters
        ----------
        element : str
            Symbol of element.
        anion : bool
            If true, then the anion of element is used. Some elements do not have
            anion information.
        cation : bool
            If true, then the cation of element is used. Some elements do not
            have cation information.

        """
        if not isinstance(element, str) or not element.isalpha():
            raise TypeError("The element argument should be all letters string.")
        if anion and cation:
            raise ValueError("Both anion and cation cannot be true.")

        data = load_slater_wfn(element, anion, cation)
        self._energy = data["energy"]
        self._potential_energy = data["potential_energy"]
        self._kinetic_energy = data["kinetic_energy"]
        self._configuration = data["configuration"]
        self._orbitals = data["orbitals"]
        self._orbitals_occupation = data["orbitals_occupation"]
        self._orbitals_basis = data["orbitals_basis"]
        self._basis_numbers = data["basis_numbers"]
        self._orbitals_exp = data["orbitals_exp"]
        self._orbitals_coeff = data["orbitals_coeff"]
        self._orbitals_energy = data["orbitals_energy"]
        self._orbitals_cusp = data["orbitals_cusp"]

    @property
    def energy(self):
        r"""Energy of atom."""
        return self._energy

    @property
    def kinetic_energy(self):
        r"""Kinetic energy of atom."""
        return self._kinetic_energy

    @property
    def potential_energy(self):
        r"""Potential energy of atom."""
        return self._potential_energy

    @property
    def configuration(self):
        r"""Return string representing the electron configuration of the atom.

        The electron configuration of the atom is written in form that writes out the
        atomic subshells with the number of electrons assigned to that atomic subshell.
        For example, Beryllium returns "1S(2)2S(2)".
        """
        return self._configuration

    @property
    def orbitals(self):
        r"""
        List of strings representing each of the atomic subshells in the electron configuration.

        For example, Beryllium returns ["1S", "2S"] in its electron configuration.
        Ordered based on "S", "P", "D", etc.
        """
        return self._orbitals

    @property
    def orbitals_occupation(self):
        r"""
        Array returning number of electrons in each of the orbitals in the electron configuration.

        For example, Beryllium returns ndarray([[2], [2]]).
        """
        return self._orbitals_occupation

    @property
    def orbitals_basis(self):
        r"""
        Return grouping of Slater-type orbital to the azimuthal quantum number ("S", "P", ...).

        Dictionary mapping type of orbital (e.g. "S", "P") to the number
        and type of the :math:`N` Slater-type orbital. For example, Helium would
        map "S" to ['2S', '1S', '1S', '1S', '2S']. This implies that all
        molecular orbitals corresponding to s-orbital will have it's radial component
        expanded in that Slater-type orbitals according to the label.
        """
        return self._orbitals_basis

    @property
    def basis_numbers(self):
        r"""
        Return type of Slater-type orbital to the type, e.g. "S".

        Dictionary mapping type of orbital (e.g. "S", "P") to array
        containing :math:`n` of the :math:`N` Slater-type orbitals. These play the
        role of the principal quantum number to each Slater-type orbitals.
        With the Helium example, "S" will map to [[2], [1], [1], [1], [2]].
        """
        return self._basis_numbers

    @property
    def orbitals_exp(self):
        r"""
        Exponent of each Slater-type orbital grouped by type of orbital.

        Dictionary mapping type of orbitals (e.g. "S", "P") to the
        exponent :math:`\alpha_j^i` of each of the :math:`M_i` Slater-type orbitals.
        """
        return self._orbitals_exp

    @property
    def orbitals_coeff(self):
        r"""
        Coefficients of each Slater-type orbital grouped by type of orbital.

        Dictionary mapping the molecular orbital (e.g. "1S", "2S", ..) to
        the coefficients :math:`c^i_j` of expansion w.r.t. the :math:`M_i` Slater-type orbital.
        """
        return self._orbitals_coeff

    @property
    def orbitals_energy(self):
        r"""Energy of each of the :math:`N` Slater-type orbital."""
        return self._orbitals_energy

    @property
    def orbitals_cusp(self):
        r"""
        Cusp values of each of the N Slater-type orbital.

        Same ordering as `orbitals`. Does not exist for Heavy atoms past Xenon.
        """
        return self._orbitals_cusp

    @staticmethod
    def radial_slater_orbital(exponent, number, points, normalized=True):
        r"""
        Compute the radial component of Slater-type orbitals on the given points.

        The radial component of the Slater-type orbital is defined as:

        .. math::
            R(r) = N r^{n-1} e^{- \alpha r}

        where,
        :math:`n` is the principal quantum number of that orbital,
        :math:`N` is the normalizing constant,
        :math:`r` is the radial point, distance to the origin, and
        :math:`\alpha` is the zeta exponent of that orbital.

        Parameters
        ----------
        exponent : ndarray, (M, 1)
            The zeta exponents :math:`\zeta` of :math:`M` Slater orbitals.
        number : ndarray, (M, 1)
            The principal quantum numbers :math:`n` of :math:`M` Slater orbitals.
        points : ndarray, (N,)
            The radial :math:`r` grid points.
        normalized : bool
            If true, adds the normalization constant :math:`N`.

        Returns
        -------
        slater : ndarray, (N, M)
            The :math:`M` Slater-type orbitals evaluated on :math:`N` grid points.

        Notes
        -----
        - The principal quantum number of all functions are stored in `basis_numbers`.

        - The alpha exponents of all functions are stored in the attribute `orbitals_exp`.

        """
        if points.ndim != 1:
            raise ValueError("The argument point should be a 1D array.")
        # compute pre-factor
        with np.errstate(divide='ignore'):
            pref = np.power(points, number - 1, dtype=np.float64).T
        # compute slater orbitals
        slater = pref * np.exp(-exponent * points).T
        # compute normalization
        if normalized:
            norm = np.power(2. * exponent, number) * \
                   np.sqrt((2. * exponent) / factorial(2. * number))
            slater *= norm.T
        return slater

    def phi_matrix(self, points, deriv=None):
        r"""
        Compute the linear combination of Slater-type orbitals on the given points.

        Each row corresponds to a point on the grid, represented as :math:`r` and
        each column is represented as a linear combination of Slater-type atomic orbitals
        of the form:

        .. math::
            \sum_{i=1}^{M} c_i R(r, n_i, \alpha_i)

        where,
        :math:`c_i` is the coefficient of the Slater-type orbital,
        :math:`\alpha_i` is the zeta exponent attached to the Slater-type orbital,
        :math:`n_i` is the principal quantum number attached to the Slater-type orbital,
        :math:`R(r, n_i, C_i)` is the radial component of the Slater-type orbital,
        :math:`M` is the number of orbitals.

        Parameters
        ----------
        points : ndarray, (N,)
            The radial grid points.
        deriv : int, optional
            If it is one, return the derivative of the slater-orbitals.
            If it is two, return the second derivative of the slater-orbitals.

        Returns
        -------
        phi_matrix : ndarray(N, K)
            The linear combination of Slater-type orbitals evaluated on the :math:`N` grid points,
            and :math:`M` is the number of atomic subshells (ignoring spin) within the electron
            configuration. The order is S orbitals, then P then D and spin is ignored.

        Notes
        -----
        - At r = 0, the derivative of slater-orbital is undefined and this function returns
          zero instead. See "derivative_radial_slater_type_orbital".

        """
        if deriv is not None:
            if deriv not in [1, 2]:
                raise ValueError(
                    f"The deriv parameter {deriv} should be either one or two. Higher-order"
                    f"is not supported."
                )

        # compute orbital composed of a linear combination of Slater
        phi_matrix = np.zeros((len(points), len(self.orbitals)))
        for index, orbital in enumerate(self.orbitals):
            exps, number = self.orbitals_exp[orbital[1]], self.basis_numbers[orbital[1]]
            if deriv == 1:
                slater = self.first_derivative_radial_slater_type_orbital(exps, number, points)
            elif deriv == 2:
                slater = self.second_derivative_radial_slater_type_orbital(exps, number, points)
            else:
                slater = self.radial_slater_orbital(exps, number, points)
            phi_matrix[:, index] = np.dot(slater, self.orbitals_coeff[orbital]).ravel()
        return phi_matrix

    def atomic_density(self, points, mode="total"):
        r"""
        Compute atomic density on the given points.

        The total density is written as a linear combination of molecular orbitals squared
        whose coefficients is the orbital occupation number of the electron configuration:

        .. math::
            \sum n_i |\phi_i(r)|^2

        where,
        :math:`n_i` is the number of electrons in the ith molecular orbital,
        :math:`\phi_i(r)` is the ith molecular orbital, whose radial component is
        a linear combination of Slater-type orbitals evaluated on the point :math:`r` and
        whose angles within the spherical coordinates are integrated.

        For core and valence density, please see more Info below.

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
            \begin{align*}
                \rho^{core}(r) &= \sum n_i (1 - e^{-|e_i - e_{homo}|^2}) |\phi_i(r)| \\
                \rho^{valence}(r) &= \sum n_i e^{-|e_i - e_{homo}|^2} |\phi_i(r)|
            \end{align*}

        where,
        :math:`e_i` is the energy of the orbital i.
        :math:`e_{HOMO}` is the energy of the highest occupying orbital.

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

    @staticmethod
    def first_derivative_radial_slater_type_orbital(exponent, number, points):
        r"""
        Compute the first derivative of the radial component of Slater-type orbital.

        The derivative of the Slater-type orbital is defined as:

        .. math::
            \frac{d R(r)}{dr} = \bigg(\frac{n-1}{r} - \alpha \bigg) N r^{n-1} e^{- \alpha r},

        where
        :math:`n` is the principal quantum number of that orbital,
        :math:`N` is the normalizing constant,
        :math:`r` is the radial point, distance to the origin, and
        :math:`\alpha` is the zeta exponent of that orbital.

        Parameters
        ----------
        exponent : ndarray, (M, 1)
            The zeta exponents of Slater orbitals.
        number : ndarray, (M, 1)
            The principle quantum numbers of Slater orbitals.
        points : ndarray, (N,)
            The radial grid points. If points contain zero, then it is undefined at those
            points and set to zero.

        Returns
        -------
        slater : ndarray, (N, M)
            First derivative of Slater-type orbitals evaluated on the :math:`N` grid points.

        """
        norm = np.power(2. * exponent, number) * np.sqrt((2. * exponent) / factorial(2. * number))
        # compute R(r) = N r^{n - 1} e^{-\alpha r}
        slater = SlaterAtoms.radial_slater_orbital(exponent, number, points, normalized=True)

        # Derivative of R(r) w.r.t. r:
        # n=1: R(r) = N e^{-\alpha r}, so dR(r)/dr = N (-\alpha) e^{-\alpha r} = -\alpha R(r)
        # n!=1: dR(r)/dr = N (-\alpha) r^{n - 1} e^{-\alpha r} + N (n - 1) r^{n - 2} e^{-\alpha r}

        # compute N (-\alpha) e^{-\alpha r} part of derivative which exists for for all n
        # -------------------------------------------------------------------------------
        deriv = np.zeros((len(points), number.shape[0]))
        deriv -= exponent.T * slater

        # compute part of the derivative which only exists for n != 1
        # -----------------------------------------------------------
        # calculate the un-normalized Slater with n - 1; i.e., r^{n - 2} e^{-\alpha r}
        slater_minus_one = SlaterAtoms.radial_slater_orbital(
            exponent, number - 1, points, normalized=False
        )
        # compute N (n - 1) r^{n - 2} e^{-\alpha r} for n != 1
        deriv_pref = norm.T * slater_minus_one * (number.T - 1.0)
        i_numb_one = np.where(number == 1)[0]
        deriv_pref[:, i_numb_one] = 0.0
        deriv += deriv_pref
        return deriv

    @staticmethod
    def second_derivative_radial_slater_type_orbital(exponent, number, points):
        r"""
        Compute the second derivative of the radial component of Slater-type orbital.

        The derivative of the Slater-type orbital is defined as:

        .. math::
            \frac{d^2 R(r)}{dr^2} = \bigg[
                \frac{(n-1)(n-2)}{r^2} - \frac{2\alpha (n-1)}{r} + \alpha^2 \bigg]
                 N r^{n-1} e^{- \alpha r},

        where
        :math:`n` is the principal quantum number of that orbital,
        :math:`N` is the normalizing constant,
        :math:`r` is the radial point, distance to the origin, and
        :math:`\alpha` is the zeta exponent of that orbital.

        Parameters
        ----------
        exponent : ndarray, (M, 1)
            The zeta exponents of Slater orbitals.
        number : ndarray, (M, 1)
            The principle quantum numbers of Slater orbitals.
        points : ndarray, (N,)
            The radial grid points. If points contain zero, then it is undefined at those
            points and set to zero.

        Returns
        -------
        slater : ndarray, (N, M)
            Second derivative of Slater-type orbitals evaluated on the :math:`N` grid points.

        """
        norm = np.power(2. * exponent, number) * np.sqrt((2. * exponent) / factorial(2. * number))
        # compute R(r) = N r^{n - 1} e^{-\alpha r}
        slater = SlaterAtoms.radial_slater_orbital(exponent, number, points, normalized=True)
        # calculate the un-normalized Slater with n - 1 and n - 2
        with np.errstate(divide='ignore'):
            slater_minus_one = SlaterAtoms.radial_slater_orbital(
                exponent, number - 1, points, normalized=False
            )
            slater_minus_two = SlaterAtoms.radial_slater_orbital(
                exponent, number - 2, points, normalized=False
            )

        # General formula for the first derivative:
        # dR(r)/dr = N (-\alpha) r^{n - 1} e^{-\alpha r} + N (n - 1) r^{n - 2} e^{-\alpha r}

        # compute N \alpha^2 e^{-\alpha r} part of derivative which exists for for all n
        # ------------------------------------------------------------------------------
        # when n=1, R(r) = N e^{-\alpha r}, so d^2R(r)/dr^2 = N \alpha^2 e^{-\alpha r}
        deriv = exponent.T**2.0 * slater

        # compute part of the derivative which only exists for n != 1
        # -----------------------------------------------------------
        # calculate 2 * N (n - 1) (-\alpha) r^{n - 2} e^{-\alpha r}
        deriv_pref = 2.0 * exponent.T * norm.T * slater_minus_one * (number.T - 1.0)
        i_numb_one = np.where(number == 1)[0]
        deriv_pref[:, i_numb_one] = 0.0
        deriv -= deriv_pref

        # compute part of the derivative which only exists for n != 1 & n != 2
        # --------------------------------------------------------------------
        # calculate N (n - 1) (n - 2) r^{n - 3} e^{-\alpha r}
        deriv_pref = norm.T * slater_minus_two * (number.T - 1.0) * (number.T - 2.0)
        i_numb_one = np.where(number == 1)[0]
        deriv_pref[:, i_numb_one] = 0.0
        deriv_pref[:, np.where(number == 2)[0]] = 0.0
        deriv += deriv_pref
        return deriv

    def positive_definite_kinetic_energy(self, points):
        r"""
        Positive definite or Lagrangian kinetic energy density.

        .. math::
            K(r) = \sum_i n_i \bigg[ \bigg(\frac{d \sum_j N_j c^i_j R_{n_j}}{dr} \bigg)^2
             + \frac{l_i (l_i + 1}{r^2} \bigg(\sum_j N_j c^i_j R_{n_j} \bigg)^2 \bigg],

        where
        :math:`n` is the principal quantum number of that orbital,
        :math:`N` is the normalizing constant, andx
        :math:`r` is the radial point, distance to the origin

        Parameters
        ----------
        points : ndarray,(N,)
            The radial grid points.

        Returns
        -------
        energy : ndarray, (N,)
            The kinetic energy on the grid points.

        Notes
        -----
        - When :math:`n=1` and :math:`r=0`, then kinetic energy is either nan or infinity.

        """
        phi_matrix = np.zeros((len(points), len(self.orbitals)))
        angular = {
            "S": 0.0, "P": 2.0, "D": 6.0, "F": 12.0
        }
        for index, orbital in enumerate(self.orbitals):
            exps, numbers = self.orbitals_exp[orbital[1]], self.basis_numbers[orbital[1]]
            # Calculate del^2 of radial component
            # derivative of the radial component
            deriv_radial = self.first_derivative_radial_slater_type_orbital(exps, numbers, points)
            phi_matrix[:, index] = np.ravel(np.dot(deriv_radial, self.orbitals_coeff[orbital])**2.0)

            # Calculate del^2 of spherical component
            norm = np.power(2. * exps, numbers) * np.sqrt((2. * exps) / factorial(2. * numbers))
            # Take unnormalized slater with number n-1, this is needed to remove divide by r^2
            slater_minus_one = SlaterAtoms.radial_slater_orbital(
                exps, numbers - 1, points, normalized=False
            )
            deriv_pref = norm.T * slater_minus_one
            # When r=0 and n = 1, then the derivative is undefined and this returns infinity or nan
            i_r_zero = np.where(np.abs(points) == 0.0)[0]
            i_numb_one = np.where(numbers[0] == 1)[0]
            indices = np.array([[x, y] for x in i_r_zero for y in i_numb_one])
            if len(indices) != 0:  # if-statement needed to remove numpy warning using list
                deriv_pref[indices] = np.inf
            # Sum the slater orbital multipled with their coefficients
            gradient_sph = (
                angular[orbital[1]] *
                np.ravel(np.dot(deriv_pref, self.orbitals_coeff[orbital]))**2.0
            )
            phi_matrix[:, index] += gradient_sph

        orb_occs = self.orbitals_occupation
        energy = np.dot(phi_matrix, orb_occs).ravel() / 2.
        return energy / (4.0 * np.pi)

    def derivative_density(self, points):
        r"""
        Return the derivative of the atomic density on a set of points.

        Parameters
        ----------
        points : ndarray(N,)
            The :math:`N` radial grid points.

        Returns
        -------
        deriv : ndarray(N,)
            The derivative of atomic density on the grid points.

        """
        factor = self.phi_matrix(points) * self.phi_matrix(points, deriv=1)
        derivative = np.dot(2. * factor, self.orbitals_occupation).ravel() / (4 * np.pi)
        return derivative

    def laplacian_of_atomic_density(self, points):
        r"""
        Return the Laplacian of the atomic density on a set of points.

        The Laplacian in radial coordinates only is:

        .. math::
            \Delta f = \frac{1}{r^2}\frac{\partial }{\partial r}\bigg(r^2 \frac{df}{dr} \bigg).

        Letting f be the atomic density we have the following equation:

        .. math::
            \Delta f = 2 \bigg[\sum n_i \big( \frac{d \phi_i}{dr}^2 +
                    \frac{2 \phi \frac{d\phi_i}{dr}}{r} +
                    \phi_i \frac{d^2 \phi_i}{dr^2} \big)
            \bigg],

        where
        :math:`\phi_i` is the ith molecular orbital with occupation number :math:`n_i`.

        Parameters
        ----------
        points : ndarray(N,)
            The :math:`N` radial grid points.

        Returns
        -------
        laplacian : ndarray(N,)
            The Laplacian of the atomic density on the grid points.

        """
        molecular_orb = self.phi_matrix(points)
        molecular_orb_deriv = self.phi_matrix(points, deriv=1)
        molecular_orb_sec_deriv = self.phi_matrix(points, deriv=2)

        # phi_i * phi_i^\prime / r
        with np.errstate(divide='ignore'):
            # Absorb phi_i / r together
            phi_i_r = np.zeros((len(points), len(self.orbitals)))
            for index, orbital in enumerate(self.orbitals):
                exps, numbers = self.orbitals_exp[orbital[1]], self.basis_numbers[orbital[1]]
                # Calculate slater divided by r
                #    Unnormalized slater with number n-1, this is needed to remove divide by r
                norm = np.power(2. * exps, numbers) * np.sqrt((2. * exps) / factorial(2. * numbers))
                slater_minus_one = SlaterAtoms.radial_slater_orbital(
                    exps, numbers - 1, points, normalized=False
                )
                slater_r = norm.T * slater_minus_one
                # When r=0 and n = 1, then slater/r is infinity.
                i_r_zero = np.where(np.abs(points) == 0.0)[0]
                i_numb_one = np.where(numbers[0] == 1)[0]
                indices = np.array([[x, y] for x in i_r_zero for y in i_numb_one])
                if len(indices) != 0:  # if-statement needed to remove numpy warning using list
                    slater_r[indices] = np.inf
                phi_i_r[:, index] += np.ravel(np.dot(slater_r, self.orbitals_coeff[orbital]))

            factor = 2.0 * phi_i_r * molecular_orb_deriv
        factor = (
            molecular_orb_deriv**2.0 + factor + molecular_orb * molecular_orb_sec_deriv
        )
        # Multiply by occupation number to get molecular orbitals.
        laplacian = np.dot(2.0 * factor, self.orbitals_occupation).ravel() / (4 * np.pi)
        return laplacian
