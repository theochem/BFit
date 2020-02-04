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
r"""
Models used for fitting.

Primarily based on Gaussian-Type Models for electronic structure theory.

Classes
-------
    AtomicGaussianDensity - Gaussian density model on one-dimensional space.
    MolecularGaussianDensity - Gaussian density model on three-dimensional space.

Notes
-----
There are two choices of Gaussian functions for both Atomic and Molecular Electronic Densities.
S-type Gaussian functions are defined to be the standard Gaussian functions.
.. math ::
        e^{-\alpha r^2}.

P-type Gaussian functions are defined as,
.. math ::
     r^2 e^{-\alpha r^2}

where :math:`\alpha` is defined to be the exponent.


Examples
--------


"""


import numpy as np

from numbers import Integral


__all__ = ["AtomicGaussianDensity", "MolecularGaussianDensity"]


class AtomicGaussianDensity:
    r"""
    Gaussian density model for modeling the radial electronic density of atoms.

    Atomic Gaussian density is a linear combination of Gaussian functions of S-type
     and P-type functions specified by coefficients :math:`c_i, d_i`, exponents `\alpha_i`,`\beta_i`
     and :math:`x_j` is the center of gaussian function. In other words,
    .. math::
        f(x) := \sum_j \sum_i c_i e^{-\alpha_i |x - x_j|^2} + d_i |x - x_j|^2 e^{-\beta |x-x_j|^2}

    Attributes
    ----------
    points : (N,) ndarray
        The grid points in a one-dimensional array.
    radii : (M, N) ndarray
        Distance of points from the center of M coordinates.
    num_s : int
        Number of S-type Gaussian basis functions.
    num_p : int
        Number of P-type Gaussian basis functions.
    nbasis : int
        Total number of basis functions (including S-type and P-type Gaussians).
    natoms : int
        Number of atom or number of centers for Gaussian function.

    Methods
    -------
    evaluate(deriv='False') :
        Return Gaussian density function on `radii` by providing coefficients and exponents.
        If `deriv` is True, then derivative with respect to the coefficient and exponent is
        also returned.

    Examples
    --------
    First define a radial grid centered around zero.
    >> point = [-0.5, -0.25, 0., 0.25, 0.5]

    Define the coordinates of each center.
    >> coordinates = np.array([-1., 1.])

    Define the Model using 5 S-Type and P-type.
    >> model = AtomicGaussianDensity(point, coord=coordinates, num_s=5, num_p=5)

    Evaluate model based on specified coefficients and exponents.
    >> coeff =
    >> exps =
    >> model.evaluate(coeff, exps)

    """

    def __init__(self, points, coord=None, num_s=1, num_p=0, normalize=False):
        r"""
        Construct class representing atomic density modeled as Gaussian functions.

        Parameters
        ----------
        points : ndarray, (N, C)
            Grid points where N is the number of points and C is the number of centers (optional).
        coord : ndarray (C,), optional
            The coordinates of gaussian basis functions centers.
            If `None`, the basis functions are placed on the origin.
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
        if coord is not None:
            if not isinstance(coord, np.ndarray) or coord.ndim != 1:
                raise ValueError("Argument coord should be a 1D numpy array.")
            if points.ndim > 1 and points.shape[1] != coord.size:
                raise ValueError("Arguments points & coord should have the same number of columns.")
        elif points.ndim > 1:
            coord = np.array([0.] * points.shape[1])
        else:
            coord = np.array([0.])
        self.coord = coord

        # compute radii (distance of points from center coord)
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

    def evaluate(self, coeffs, expons, deriv=False):
        """Compute linear combination of Gaussian basis & its derivatives on the grid points.

        .. math::
            f(x) := \sum_i \sum_j c_i e^{-\alpha_i |x - x_j|^2} + d_i |x-x_j|^2 e^{-\beta |x-x_j|^2}
        where :math:`c_i, d_i \alpha_i, \beta_i, x_j` is the coefficient of S-type,
            coefficient of P-type, exponents of S and P-type and atom coordinates, respectively.

        Parameters
        ----------
        coeffs : ndarray, (`nbasis`,)
            The coefficients of `num_s` s-type Gaussian basis functions followed by the
            coefficients of `num_p` p-type Gaussian basis functions.
        expons : ndarray, (`nbasis`,)
            The exponents of `num_s` s-type Gaussian basis functions followed by the
            exponents of `num_p` p-type Gaussian basis functions.
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
            raise ValueError("Argument coeffs should have size {0}.".format(self.nbasis))

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
        """Compute linear combination of s-type Gaussian basis & its derivative on the grid points.

        Parameters
        ----------
        matrix : ndarray, (N, M)
             The exp(-a * r**2) array evaluated on grid points for each exponent.
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
        """Compute linear combination of P-type Gaussian basis & its derivative on the grid points.

        Parameters
        ----------
        matrix : ndarray, (N, M)
             The exp(-a * r**2) array evaluated on grid points for each exponent.
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
    """Molecular Atom-Centered Gaussian Density Model.

    Attributes
    ----------
    points : (N, 3) ndarray
        The grid points in a two-dimensional array, where N is the number of points.
    radii : (M, N) ndarray
        Distance of points from the center of M coordinates.
    num_s : int
        Number of S-type Gaussian basis functions.
    num_p : int
        Number of P-type Gaussian basis functions.
    nbasis : int
        Total number of basis functions (including S-type and P-type Gaussians).
    natoms : int
        Number of atom or number of centers for Gaussian function.

    Methods
    -------

    """

    def __init__(self, points, coords, basis, normalize=False):
        """
        Parameters
        ----------
        points : ndarray, (N, 3)
            The grid points, where N is the number of grid points.
        coords : ndarray, (M, 3)
            The atomic coordinates (M centers) on which Gaussian basis are centered.
        basis : ndarray, (M, 2)
            The number of s-type & p-type Gaussian basis functions placed on each center.
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
        """The grid points."""
        return self._points

    @property
    def nbasis(self):
        """The total number of Gaussian basis functions."""
        return np.sum(self._basis)

    @property
    def radii(self):
        """The distance of grid points from center of each basis function."""
        return self._radii

    @property
    def natoms(self):
        """Number of basis functions centers."""
        return len(self._basis)

    @property
    def prefactor(self):
        """The pre-factor of Gaussian basis functions"""
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
            raise ValueError("The {0} is invalid for {1} basis.".format(index, self.nbasis))
        # compute the number of basis on each center
        nbasis = np.sum(self._basis, axis=1)
        # get the center to which the basis function belongs
        index = np.where(np.cumsum(nbasis) >= index + 1)[0][0]
        return index

    def evaluate(self, coeffs, expons, deriv=False):
        """Compute linear combination of Gaussian basis & its derivatives on the grid points.

        .. math::

        Parameters
        ----------
        coeffs : ndarray, (`nbasis`,)
            The coefficients of `num_s` s-type Gaussian basis functions followed by the
            coefficients of `num_p` p-type Gaussian basis functions.
        expons : ndarray, (`nbasis`,)
            The exponents of `num_s` s-type Gaussian basis functions followed by the
            exponents of `num_p` p-type Gaussian basis functions.
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
            raise ValueError("Arguments coeffs & expons shape != ({0},)".format(self.nbasis))
        
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
