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
r"""Models used for fitting."""

from numbers import Integral

import numpy as np

__all__ = ["AtomicGaussianDensity", "MolecularGaussianDensity"]


class AtomicGaussianDensity:
    r"""
    Gaussian density model for modeling the electronic density of a single atom.

    Atomic Gaussian density is a linear combination of Gaussian functions of S-type
    and p-type functions:

    .. math::
        f(x) := \sum_i c_i e^{-\alpha_i |x - c|^2} + \sum_j d_j |x - c|^2 e^{-\beta_j |x - c|^2}

    where
    :math:`c_i, d_i` are the coefficients of s-type and p-type Gaussian functions,
    :math:`\alpha_i, \beta_j` are teh exponents of the s-type and p-type Gaussian functions,
    :math:`c` is the center of all Gaussian functions.
    :math:`x` is the real coordinates, can be multi-dimensional.

    """

    def __init__(self, points, center=None, num_s=1, num_p=0, normalize=False):
        r"""
        Construct class representing atomic density modeled as Gaussian functions.

        Parameters
        ----------
        points : ndarray, (N, D)
            Grid points where N is the number of points and D is the number of dimensions.
        center : ndarray (D,), optional
            The D-dimensional coordinates of the single center.
            If `None`, then the center is the origin of all zeros.
        num_s : int, optional
             Number of s-type Gaussian basis functions.
        num_p : int, optional
             Number of p-type Gaussian basis functions.
        normalize : bool, optional
            Whether to normalize Gaussian basis functions.

        """
        if not isinstance(points, np.ndarray):
            raise TypeError("Argument points should be a numpy array.")
        if not isinstance(num_s, Integral) or num_s < 0:
            raise TypeError("Argument num_s should be a positive integer.")
        if not isinstance(num_p, Integral) or num_p < 0:
            raise TypeError("Argument num_p should be a positive integer.")
        if num_s + num_p == 0:
            raise ValueError("Arguments num_s & num_p cannot both be zero!")

        # check & assign coordinates.
        if center is not None:
            if not isinstance(center, np.ndarray) or center.ndim != 1:
                raise ValueError("Argument center should be a 1D numpy array.")
            if points.ndim > 1 and points.shape[1] != center.size:
                raise ValueError("Points & center should have the same number of columns.")
        elif points.ndim > 1:
            center = np.array([0.] * points.shape[1])
        else:
            center = np.array([0.])
        self.coord = center

        # compute radii (distance of points from center center)
        if points.ndim > 1:
            radii = np.linalg.norm(points - self.coord, axis=1)
        else:
            radii = np.abs(points - self.coord)
        self._radii = np.ravel(radii)

        self._points = points
        self.ns = num_s
        self.np = num_p
        self.normalized = normalize

    @property
    def points(self):
        """Return the grid points."""
        return self._points

    @property
    def radii(self):
        """Return the distance of grid points from center of Gaussian(s)."""
        return self._radii

    @property
    def num_s(self):
        """Return the number of s-type Gaussian basis functions."""
        return self.ns

    @property
    def num_p(self):
        """Return the number of p-type Gaussian basis functions."""
        return self.np

    @property
    def nbasis(self):
        """Return the total number of Gaussian basis functions."""
        return self.ns + self.np

    @property
    def natoms(self):
        """Return the number of basis functions centers."""
        return 1

    @property
    def prefactor(self):
        r"""Obtain list of exponents for the prefactors."""
        return np.array([1.5] * self.ns + [2.5] * self.np)

    def change_numb_s_and_numb_p(self, new_s, new_p):
        r"""
        Change the number of s-type and p-type Gaussians.

        Parameters
        ----------
        new_s : int
            New number of s-type Gaussians.
        new_p : int
            New number of p-type Gaussians.

        """
        if not isinstance(new_s, int):
            raise TypeError(f"New number of s-type {new_s} should be of type int.")
        if not isinstance(new_p, int):
            raise TypeError(f"New number of p-type {new_p} should be of type int.")
        self.ns = new_s
        self.np = new_p

    def evaluate(self, coeffs, expons, deriv=False):
        r"""
        Compute linear combination of Gaussian basis & its derivatives on the grid points.

        .. math::
            f(x):= \sum_i c_i e^{-\alpha_i |x - c|^2} + \sum_j d_j |x - c|^2 e^{-\beta_j |x - c|^2}

        where
        :math:`c_i, d_i` are the coefficients of s-type and p-type Gaussian functions,
        :math:`\alpha_i, \beta_j` are teh exponents of the s-type and p-type Gaussian functions,
        :math:`c` is the center of all Gaussian functions.
        :math:`x` is the real coordinates, can be multi-dimensional.

        Parameters
        ----------
        coeffs : ndarray(`nbasis`,)
            The coefficients :math:`c_i` of `num_s` s-type Gaussian basis functions
            followed by the coefficients :math:`d_j` of `num_p` p-type Gaussian basis functions.
        expons : ndarray(`nbasis`,)
            The exponents :math:`\alpha_i` of `num_s` s-type Gaussian basis functions
            followed by the exponents :math:`\beta_j` of `num_p` p-type Gaussian basis functions.
        deriv : bool, optional
            Whether to compute derivative of Gaussian basis functions w.r.t. coefficients &
            exponents.

        Returns
        -------
        g : ndarray, (N,)
            The linear combination of Gaussian basis functions evaluated on the grid points.
        dg : ndarray, (N, 2 * `nbasis`)
            The derivative of a linear combination of Gaussian basis functions w.r.t. coefficients
            & exponents, respectively, evaluated on the grid points. Only returned if `deriv=True`.

        """
        if coeffs.ndim != 1 or expons.ndim != 1:
            raise ValueError("Arguments coeffs and expons should be 1D arrays.")
        if coeffs.size != expons.size:
            raise ValueError("Arguments coeffs and expons should have the same length.")
        if coeffs.size != self.nbasis:
            raise ValueError(f"Argument coeffs should have size {self.nbasis}.")

        # evaluate all Gaussian basis on the grid, i.e., exp(-a * r**2)
        matrix = np.exp(-expons[None, :] * np.power(self.radii, 2)[:, None])

        # compute linear combination of Gaussian basis
        if self.np == 0:
            # only s-type Gaussian basis functions
            return self._eval_s(matrix, coeffs, expons, deriv)
        elif self.ns == 0:
            # only p-type Gaussian basis functions
            return self._eval_p(matrix, coeffs, expons, deriv)
        else:
            # both s-type & p-type Gaussian basis functions
            gs = self._eval_s(matrix[:, :self.ns], coeffs[:self.ns], expons[:self.ns], deriv)
            gp = self._eval_p(matrix[:, self.ns:], coeffs[self.ns:], expons[self.ns:], deriv)
            if deriv:
                # split derivatives w.r.t. coeffs & expons
                d_coeffs = np.concatenate((gs[1][:, :self.ns], gp[1][:, :self.np]), axis=1)
                d_expons = np.concatenate((gs[1][:, self.ns:], gp[1][:, self.np:]), axis=1)
                return gs[0] + gp[0], np.concatenate((d_coeffs, d_expons), axis=1)
            return gs + gp

    def _eval_s(self, matrix, coeffs, expons, deriv):
        r"""
        Compute linear combination of s-type Gaussian basis & its derivative on the grid points.

        Parameters
        ----------
        matrix : ndarray, (N, M)
             The exp(-\alpha_i * r**2) array evaluated on grid points for each exponent.
        coeffs : ndarray, (M,)
            The coefficients of Gaussian basis functions.
        expons : ndarray, (M,)
            The exponents of Gaussian basis functions.
        deriv : bool, optional
            Whether to compute derivative of Gaussian basis functions w.r.t. coefficients &
            exponents.

        Returns
        -------
        g : ndarray, (N,)
            The linear combination of s-type Gaussian basis functions evaluated on the grid points.
        dg : ndarray, (N, 2*M)
            The derivative of linear combination of s-type Gaussian basis functions w.r.t.
            coefficients (the 1st M columns) & exponents (the 2nd M columns) evaluated on the
            grid points. Only returned if `deriv=True`.

        """
        # normalize Gaussian basis
        if self.normalized:
            matrix = matrix * (expons[None, :] / np.pi) ** 1.5
        # make linear combination of Gaussian basis on the grid
        g = np.dot(matrix, coeffs)

        # compute derivatives
        if deriv:
            dg = np.zeros((len(self.radii), 2 * coeffs.size))
            # derivative w.r.t. coefficients
            dg[:, :coeffs.size] = matrix
            # derivative w.r.t. exponents
            dg[:, coeffs.size:] = - matrix * np.power(self.radii, 2)[:, None] * coeffs[None, :]
            if self.normalized:
                matrix = np.exp(-expons[None, :] * np.power(self.radii, 2)[:, None])
                dg[:, coeffs.size:] += 1.5 * matrix * (coeffs * expons**0.5)[None, :] / np.pi**1.5
            return g, dg
        return g

    def _eval_p(self, matrix, coeffs, expons, deriv):
        """Compute linear combination of p-type Gaussian basis & its derivative on the grid points.

        Parameters
        ----------
        matrix : ndarray, (N, M)
             The exp(-beta_i * r**2) array evaluated on grid points for each exponent.
        coeffs : ndarray, (M,)
            The coefficients of Gaussian basis functions.
        expons : ndarray, (M,)
            The exponents of Gaussian basis functions.
        deriv : bool, optional
            Whether to compute derivative of Gaussian basis functions w.r.t. coefficients &
            exponents.

        Returns
        -------
        g : ndarray, (N,)
            The linear combination of p-type Gaussian basis functions evaluated on the grid points.
        dg : ndarray, (N, 2*M)
            The derivative of linear combination of p-type Gaussian basis functions w.r.t.
            coefficients (the 1st M columns) & exponents (the 2nd M columns) evaluated on the
            grid points. Only returned if `deriv=True`.

        """
        # multiply r**2 with the evaluated Gaussian basis, i.e., r**2 * exp(-a * r**2)
        matrix = matrix * np.power(self.radii, 2)[:, None]

        if not self.normalized:
            # linear combination of p-basis is the same as s-basis with an extra r**2
            return self._eval_s(matrix, coeffs, expons, deriv)

        # normalize Gaussian basis
        matrix = matrix * (expons[None, :]**2.5 / np.pi**1.5) / 1.5
        # make linear combination of Gaussian basis on the grid
        g = np.dot(matrix, coeffs)
        if deriv:
            dg = np.zeros((len(self.radii), 2 * coeffs.size))
            # derivative w.r.t. coefficients
            dg[:, :coeffs.size] = matrix
            # derivative w.r.t. exponents
            dg[:, coeffs.size:] = - matrix * np.power(self.radii, 2)[:, None] * coeffs[None, :]
            matrix = np.exp(-expons[None, :] * np.power(self.radii, 2)[:, None])
            matrix = matrix * np.power(self.radii, 2)[:, None]
            dg[:, coeffs.size:] += 5 * matrix * (coeffs * expons**1.5)[None, :] / (3 * np.pi**1.5)
            return g, dg
        return g


class MolecularGaussianDensity:
    r"""
    Molecular Atom-Centered Gaussian Density Model.

    The Molecular Gaussian Density model is based on multiple centers each associated with a
    Gaussian density model (s or p-type) of any dimension.

    .. math::
        f(x) := \sum_j \bigg[ \sum_{i =1}^{M^s_j} c_{ji} e^{-\alpha_{ji} |x - m_j|^2} +
                         \sum_{i=1}^{M_j^p}d_{ji} |x - m_j|^2 e^{-\beta_{ji} |x - m_j|^2} \bigg]

    where
    :math:`c_{ji}, d_{ji}` are the ith coefficients of s-type and p-type functions of the
    jth center, :math:`\alpha_{ji}, \beta_{ji}` are the ith exponents of S-type and P-type
    functions of the jth center, :math:`M_j^s, M_j^p` is the total number of s-type or p-type
    Gaussians functions of the jth center respectively,
    :math:`m_j` is the coordinate of the jth center, and
    :math:`x` is the real coordinates of the point. It can be of any dimension.
    """

    def __init__(self, points, coords, basis, normalize=False):
        """
        Construct the MolecularGaussianDensity class.

        Parameters
        ----------
        points : ndarray, (N, D)
            The grid points, where N is the number of grid points and D is the dimension.
        coords : ndarray, (M, D)
            The atomic coordinates (M centers) on which Gaussian basis are centered.
        basis : ndarray, (M, 2)
            The number of S-type & P-type Gaussian basis functions placed on each center.
        normalize : bool, optional
            Whether to normalize Gaussian basis functions.

        """
        # check arguments
        if not isinstance(coords, np.ndarray) or coords.ndim != 2:
            raise ValueError("Argument coords should be a 2D numpy array.")
        if basis.ndim != 2 or basis.shape[1] != 2:
            raise ValueError("Argument basis should be a 2D array with 2 columns.")
        if len(coords) != len(basis):
            raise ValueError("Argument coords & basis should represent the same number of atoms.")
        if points.ndim > 1 and points.shape[1] != coords.shape[1]:
            raise ValueError("Arguments points & coords should have the same number of columns.")

        self._points = points
        self._basis = basis
        # place a GaussianModel on each center
        self.center = []
        self._radii = []
        for i, b in enumerate(basis):
            # get the center of Gaussian basis functions
            self.center.append(AtomicGaussianDensity(points, coords[i], b[0], b[1], normalize))
            self._radii.append(self.center[-1].radii)
        self._radii = np.array(self._radii)

    @property
    def points(self):
        """Get grid points."""
        return self._points

    @property
    def nbasis(self):
        """Get the total number of Gaussian basis functions."""
        return np.sum(self._basis)

    @property
    def radii(self):
        """Get the distance of grid points from center of each basis function."""
        return self._radii

    @property
    def natoms(self):
        """Get number of basis functions centers."""
        return len(self._basis)

    @property
    def prefactor(self):
        """
        Get the pre-factor of Gaussian basis functions to make it normalized.

        Only used if attribute `normalize` is true.
        """
        return np.concatenate([center.prefactor for center in self.center])

    def assign_basis_to_center(self, index):
        """Assign the Gaussian basis function to the atomic center.

        Parameters
        ----------
        index : int
            The index of Gaussian basis function.

        Returns
        -------
        index : int
            The index of atomic center.

        """
        if index >= self.nbasis:
            raise ValueError(f"The {index} is invalid for {self.nbasis} basis.")
        # compute the number of basis on each center
        nbasis = np.sum(self._basis, axis=1)
        # get the center to which the basis function belongs
        index = np.where(np.cumsum(nbasis) >= index + 1)[0][0]
        return index

    def evaluate(self, coeffs, expons, deriv=False):
        r"""
        Compute linear combination of Gaussian basis & its derivatives on the grid points.

        The Molecular Gaussian is defined to be:

         .. math::
            f(x) := \sum_j \bigg[ \sum_{i =1}^{M^s_j} c_{ji} e^{-\alpha_{ji} |x - m_j|^2} +
                    \sum_{i=1}^{M_j^p}d_{ji} |x - m_j|^2 e^{-\beta_{ji} |x - m_j|^2} \bigg]

        where
        :math:`c_{ji}, d_{ji}` are the ith coefficients of s-type and p-type functions of the
        jth center, :math:`\alpha_{ji}, \beta_{ji}` are the ith exponents of S-type and P-type
        functions of the jth center, :math:`M_j^s, M_j^p` is the total number of s-type or p-type
        Gaussians functions of the jth center respectively,
        :math:`m_j` is the coordinate of the jth center, and
        :math:`x` is the real coordinates of the point. It can be of any dimension.

        Parameters
        ----------
        coeffs : ndarray, (`nbasis`,)
            The coefficients of `num_s` s-type Gaussian basis functions followed by the
            coefficients of `num_p` p-type Gaussian basis functions for an atom, then repeat
            for the next atom.
        expons : ndarray, (`nbasis`,)
            The exponents of `num_s` s-type Gaussian basis functions followed by the
            exponents of `num_p` p-type Gaussian basis functions for an atom, then repeat
            for the next atom.
        deriv : bool, optional
            Whether to compute derivative of Gaussian basis functions w.r.t. coefficients &
            exponents.

        Returns
        -------
        g : ndarray, (N,)
            The linear combination of Gaussian basis functions evaluated on the grid points.
        dg : ndarray, (N, `nbasis`)
            The derivative of linear combination of Gaussian basis functions w.r.t. coefficients
            & exponents, respectively, evaluated on the grid points. Only returned if `deriv=True`.

        """
        if coeffs.ndim != 1 or expons.ndim != 1:
            raise ValueError("Arguments coeffs & expons should be 1D arrays.")
        if coeffs.size != self.nbasis or expons.size != self.nbasis:
            raise ValueError(f"Arguments coeffs & expons shape != ({self.nbasis},)")

        # assign arrays
        total_g = np.zeros(len(self.points))
        if deriv:
            total_dg = np.zeros((len(self.points), 2 * self.nbasis))
        # compute contribution of each center
        count = 0
        for center in self.center:
            # get coeffs & expons of center
            cs = coeffs[count: count + center.nbasis]
            es = expons[count: count + center.nbasis]
            if deriv:
                # compute linear combination of gaussian placed on center & its derivatives
                g, dg = center.evaluate(cs, es, deriv)
                # split derivatives w.r.t. coeffs & expons
                dg_c = dg[:, :center.nbasis]
                dg_e = dg[:, center.nbasis:]
                # add contributions to the total array
                total_g += g
                total_dg[:, count: count + center.nbasis] = dg_c
                total_dg[:, self.nbasis + count: self.nbasis + count + center.nbasis] = dg_e
            else:
                # compute linear combination of gaussian placed on center
                total_g += center.evaluate(cs, es, deriv)
            count += center.nbasis
        if deriv:
            return total_g, total_dg
        return total_g
