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
"""
Grid Module - Model the grid and integration methods of the model and true probability distributions.

Methods
-------
There are three classes:

ClenshawRadialGrid -
    Models a One-dimensional grid via Clenshaw-Curtis pattern.
    Integration is done via 'np.trapz' method (spherical coordinates, optional).
    Intended for Atomic fitting.

UniformRadialGrid -
    Uniform (equal spacing), one-dimensional grid.
    Integration is done via 'np.trapz' method (spherical coordinates, optional).
    Intended for Atomic fitting.

CubicGrid -
    Uniform (equal spacing), three-dimensional grid.
    Integration is done via Riemannian sums.
    Intended for Molecular fitting.

"""


import numpy as np


__all__ = ["ClenshawRadialGrid", "UniformRadialGrid", "CubicGrid"]


class _BaseRadialGrid(object):
    r"""
    Radial Grid Base Class.

    Attributes
    ----------
    points : ndarray, (N,)
        The radial grid points of a one-dimensional grid with N points.
    spherical : bool
        If true, then trapezoidal integration is done spherically (ie with a factor of :math:`4 \pi r^2`).

    Methods
    -------
    integrate(arr)
        Integrate array `arr` defined over "point" array using trapezoidal integration.

    """

    def __init__(self, points, spherical=True):
        """
        Construct BaseRadialGrid object.

        Parameters
        ----------
        points : ndarray, (N,)
            The radial grid points.
        spherical : bool, optional
            If `True`, the spherical integration of function is computed.

        """
        if not isinstance(points, np.ndarray) or points.ndim != 1:
            raise TypeError("Argument points should be a 1D numpy array.")
        self._points = np.ravel(points)
        self._spherical = spherical

    @property
    def points(self):
        """Radial grid points."""
        return self._points

    @property
    def spherical(self):
        """Whether to perform spherical integration."""
        return self._spherical

    def __len__(self):
        """Return number of grid points."""
        return self._points.shape[0]

    def integrate(self, arr):
        r"""Compute trapezoidal integration of a function evaluated on the radial grid points.

        .. math:: \int p * f(r) dr

        where :math:'f(r)' is the integrand and :math:`p=4 \times \pi` if `spherical=True`,
        otherwise :math:`p=1.0`.

        Parameters
        ----------
        arr : ndarray
            The integrand evaluated on the radial grid points.

        Returns
        -------
        value : float
            The value of integral.

        """
        if arr.shape != self.points.shape:
            raise ValueError("The argument arr should have {0} shape!".format(self.points.shape))
        if self._spherical:
            value = 4. * np.pi * np.trapz(y=self.points**2 * arr, x=self.points)
        else:
            value = np.trapz(y=arr, x=self.points)
        return value


class UniformRadialGrid(_BaseRadialGrid):
    r"""Uniformly Distributed Radial Grid Class.

    The grid points are equally-spaced in :math:`[0, max_points)` interval.

    Attributes
    ----------
    points : ndarray, (N,)
        The radial grid points of a one-dimensional grid with N points.
    spherical : bool
        If true, then trapezoidal integration is done spherically (ie with a factor of :math:`4 \pi r^2`).

    Methods
    -------
    integrate(arr)
        Integrate array `arr` defined over "point" array using trapezoidal integration.

    """

    def __init__(self, num_pts, min_radii=0., max_radii=100., spherical=True):
        """
        Construct the UniformRadialGrid object.

        Parameters
        ----------
        num_pts : int
            The number of grid points.
        min_radii : float, optional
            The smallest radial grid point.
        max_radii : float, optional
            The largest radial grid point.
        spherical : bool, optional
            If `True`, the spherical integration of function is computed.

        """
        if not isinstance(num_pts, int) or num_pts <= 0:
            raise TypeError("Argument num_pts should be a number.")
        if not isinstance(min_radii, (int, float)):
            raise TypeError("Argument min_radii should be a number.")
        if not isinstance(max_radii, (int, float)):
            raise TypeError("Argument max_radii should be a positive number.")
        if max_radii <= min_radii:
            raise ValueError("The max_radii should be greater than the min_radii.")

        # compute points
        points = np.linspace(start=min_radii, stop=max_radii, num=num_pts)
        super(UniformRadialGrid, self).__init__(points, spherical)


class ClenshawRadialGrid(_BaseRadialGrid):
    r"""Clenshaw-Curtis Radial Grid Class.

    The Clenshaw-Curtis grid places more points closer to the origin of the interval :math:`[0, \inf).`
    It is defined as follows. Let :math:`Z, m, n` be the atomic number, number of points near origin,
    and the number of points far from the origin, respectively.

    Then each point :math:`r_p` of the Clenshaw radial grid is defined as:

    .. math::
        \begin{eqnarray}
            r_p = \frac{1}{2Z} \bigg(1 - \cos\bigg(\frac{\pi p}{400} \bigg)\bigg)  & p = 0, 1, \cdots, m - 1 \\
            r_p = 25 \bigg(1 - \cos\bigg(\frac{\pi p}{600} \bigg)\bigg) & p = 0, 1, \cdocts, n - 1\\
        \end{eqnarray}

    Attributes
    ----------
    points : ndarray, (N,)
        The radial grid points of a one-dimensional grid with N points.
    atomic_number :
        Return the atomic number.
    spherical : bool
        If true, then trapezoidal integration is done spherically (ie with a factor of :math:`4 \pi r^2`).

    Methods
    -------
    integrate(arr)
        Integrate array `arr` defined over "point" array using trapezoidal integration.

    """

    def __init__(self, atomic_number, num_core_pts, num_diffuse_pts, extra_pts=None, spherical=True):
        r"""
        Construct ClenshawRadialGrid grid object.

        Parameters
        ----------
        atomic_number : int
            The atomic number of the atom for which the grid is generated.
        num_core_pts : int
            The number of points near the origin/core region.
        num_diffuse_pts : int
            The number of points far from the origin/core region.
        extra_pts : list
            Additional points to be added to the grid, commonly points far away from origin.
        spherical : bool, optional
            If `True`, the spherical integration of function is computed.

        """
        if not isinstance(atomic_number, int) or atomic_number <= 0:
            raise TypeError("Argument atomic_number should be a positive integer.")
        if not isinstance(num_core_pts, int) or num_core_pts < 0:
            raise TypeError("Argument numb_core_pts should be a non-negative integer.")
        if not isinstance(num_diffuse_pts, int) or num_diffuse_pts < 0:
            raise TypeError("Argument num_diffuse_pts should be a non-negative integer.")

        self._atomic_number = atomic_number

        # compute core and diffuse points
        core_points = self._get_points(num_core_pts, mode="core")
        diff_points = self._get_points(num_diffuse_pts, mode="diffuse")

        # put all points together (0.0 is also contained in diff_points, so it should be removed)
        if extra_pts:
            # check extra points
            if not hasattr(extra_pts, '__iter__') or isinstance(extra_pts, str):
                raise TypeError("Argument extra_pts should be an iterable.")
            points = np.concatenate((core_points, diff_points[1:], extra_pts))
        else:
            points = np.concatenate((core_points, diff_points[1:]))

        super(ClenshawRadialGrid, self).__init__(np.sort(points), spherical)

    @property
    def atomic_number(self):
        """Return the atomic number."""
        return self._atomic_number

    def _get_points(self, num_pts, mode="core"):
        r"""Generate radial points on [0, inf) based on Clenshaw-Curtis grid.

        The "core" points are concentrated near the origin based on:

        .. math:: r_p = 25 (1 - cos(\frac{\pi p}{2N})) for p =0,1..N-1

        The "diffuse" points are concentrated away from the origin based on:

        .. math:: r_p = \frac{1}[2Z} (1 - cos(\frac{\pi p}{2N})) for p=0,1...N-1,

        where :math:`Z` is the atomic number and :math:`N` is the number of points.

        Parameters
        ----------
        num_pts : int
            The number of points.
        mode : str, optional
            If "core", the points are placed closer to the origin. If "diffuse", the points are
            placed far away from origin.

        Returns
        -------
        points : ndarray, (N,)
            The 1D array of grid points.

        """
        if mode.lower() == "core":
            points = 1. - np.cos(0.5 * np.pi * np.arange(0, num_pts) / num_pts)
            points /= 2 * self._atomic_number
        elif mode.lower() == "diffuse":
            points = 25. * (1. - np.cos(0.5 * np.pi * np.arange(0, num_pts) / num_pts))
        else:
            raise ValueError("Arguments mode={0} is not recognized!".format(mode.lower()))
        return points


class CubicGrid(object):
    r"""
    Equally-Spaced 3D Cubic Grid Class.

    Attributes
    ----------
    points : ndarray, (N, 3)
        The three-dimensional array containing the `N` grid points that are uniform.
    step : float
        The positive number representing the step-size of any two consequent grid points.

    Methods
    -------
    integrate(arr) :
        Integrate an array `arr` defined over the `points` using Riemannian sum.

    """

    def __init__(self, smallest_pnt, largest_pnt, step_size):
        """
        Construct the CubicGrid object.

        Parameters
        ----------
        smallest_pnt : float
            The smallest point on any axis in the 3D cubic grid.
        largest_pnt : float
            The largest point on any axis in the 3D cubic grid.
        step_size : float
            The step-size between two consecutive points on any axis in the 3D cubic grid.

        """
        if not isinstance(smallest_pnt, (int, float)):
            raise TypeError("Argument smallest_pt should be a positive number.")
        if not isinstance(largest_pnt, (int, float)):
            raise TypeError("Argument largest_pnt should be a positive number.")
        if not isinstance(step_size, (int, float)) or step_size <= 0:
            raise TypeError("The argument step_size should be a positive number")
        if largest_pnt <= smallest_pnt:
            raise ValueError("The largest_pnt should be greater than the smallest_pnt.")

        # compute points in one dimension
        points_1d = np.arange(smallest_pnt, largest_pnt + step_size, step_size)
        npoints = points_1d.size
        points = np.zeros((npoints**3, 3))
        # assign x, y & z coordinates
        points[:, 0] = np.repeat(points_1d, npoints**2)
        points[:, 1] = np.tile(np.repeat(points_1d, npoints), npoints)
        points[:, 2] = np.tile(points_1d, npoints**2)
        self._points = np.array(points)
        self._step = step_size

    @property
    def points(self):
        """Return cubic grid points."""
        return self._points

    @property
    def step(self):
        """Return the step size between to consecutive points."""
        return self._step

    def __len__(self):
        """Return the number of grid points."""
        return self._points.shape[0]

    def integrate(self, arr):
        r"""Compute the integral of a function evaluated on the grid points based on Riemann sums.

        .. math:: \int\int\int f(x, y, z) dx dy dz

        where :math:'f(r)' is the integrand.

        Parameters
        ----------
        arr : ndarray
            The integrand evaluated on the grid points.

        Returns
        -------
        value : float
            The value of integral.

        """
        if arr.shape != (len(self),):
            raise ValueError("Argument arr should have ({0},) shape.".format(len(self)))
        value = np.power(self._step, 3) * np.sum(arr)
        return value
