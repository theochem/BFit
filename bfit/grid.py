# -*- coding: utf-8 -*-
# BFit - python program that fits a convex sum of
# positive basis functions to any probability distribution. .
#
# Copyright (C) 2020 The BFit Development Team.
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
"""Grid Module For Integration"""

import numpy as np


__all__ = ["ClenshawRadialGrid", "UniformRadialGrid", "CubicGrid"]


class _BaseRadialGrid:
    r"""Radial Grid Base Class."""

    def __init__(self, points):
        """
        Construct BaseRadialGrid object.

        Parameters
        ----------
        points : ndarray, (N,)
            The radial grid points.

        """
        if not isinstance(points, np.ndarray) or points.ndim != 1:
            raise TypeError("Argument points should be a 1D numpy array.")
        self._points = np.ravel(points)

    @property
    def points(self):
        """Radial grid points."""
        return self._points

    def __len__(self):
        """Return number of grid points."""
        return self._points.shape[0]

    def integrate(self, arr):
        r"""Compute trapezoidal integration of a function evaluated on the radial grid points.

        :math:`\int f(r) dr`, where :math:'f(r)' is the integrand.

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
        value = np.trapz(y=arr, x=self.points)
        return value


class UniformRadialGrid(_BaseRadialGrid):
    r"""
    Uniformly Distributed Radial Grid Class.

    The grid points are equally-spaced in :math:`[0, max_points)` interval.
    """

    def __init__(self, num_pts, min_radii=0., max_radii=100.):
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

    """

    def __init__(self, atomic_number, num_core_pts, num_diffuse_pts, extra_pts=None):
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

        super(ClenshawRadialGrid, self).__init__(np.sort(points))

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
            points = 1. - np.cos(0.5 * np.pi * np.arange(0., num_pts) / num_pts)
            points /= 2 * self._atomic_number
        elif mode.lower() == "diffuse":
            points = 25. * (1. - np.cos(0.5 * np.pi * np.arange(0., num_pts) / num_pts))
        else:
            raise ValueError("Arguments mode={0} is not recognized!".format(mode.lower()))
        return points


class CubicGrid:
    r"""Equally-Spaced 3D Cubic Grid Class."""

    def __init__(self, origin, axes, shape):
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
        # TODO: Add raise error for Type and Values here for origin, axes.
        self._axes = axes
        self._origin = origin
        dim = self._origin.size
        # Make an array to store coordinates of grid points
        self._points = np.zeros((np.prod(shape), dim))
        coords = np.array(
            np.meshgrid(
                np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])
            )
        )
        coords = np.swapaxes(coords, 1, 2)
        coords = coords.reshape(3, -1)
        points = coords.T.dot(self._axes) + origin
        # assign the weights
        weights = self._choose_weight_scheme(shape)
        self._weights = weights
        self._points = np.array(points)

    @property
    def axes(self):
        return self._axes

    @classmethod
    def from_molecule(
            cls,
            atcorenums,
            atcoords,
            spacing=0.2,
            extension=5.0,
            rotate=True,
    ):
        r"""
        Construct a uniform grid given the molecular pseudo-numbers and coordinates.

        Parameters
        ----------
        atcorenums : np.ndarray, shape (M,)
            Pseudo-number of :math:`M` atoms in the molecule.
        atcoords : np.ndarray, shape (M, 3)
            Cartesian coordinates of :math:`M` atoms in the molecule.
        spacing : float, optional
            Increment between grid points along :math:`x`, :math:`y`, and :math:`z` direction.
        extension : float, optional
            The extension of the length of the cube on each side of the molecule.
        rotate : bool, optional
            When True, the molecule is rotated so the axes of the cube file are
            aligned with the principle axes of rotation of the molecule.
            If False, generates axes based on the x,y,z-axis and the spacing parameter, and
            the origin is defined by the maximum/minimum of the atomic coordinates.
        """
        # calculate center of mass of the nuclear charges:
        totz = np.sum(atcorenums)
        com = np.dot(atcorenums, atcoords) / totz
        # Determine best axes and coordinates to calculate the lower and upper bound of grid.
        if rotate:
            # calculate moment of inertia tensor:
            itensor = np.zeros([3, 3])
            for i in range(atcorenums.shape[0]):
                xyz = atcoords[i] - com
                r = np.linalg.norm(xyz) ** 2.0
                tempitens = np.diag([r, r, r])
                tempitens -= np.outer(xyz.T, xyz)
                itensor += atcorenums[i] * tempitens
            # Eigenvectors define the new axes of the grid with spacing
            _, v = np.linalg.eigh(itensor)
            # Project the coordinates of atoms centered at the center of mass to the eigenvectors
            new_coordinates = np.dot((atcoords - com), v)
            axes = spacing * v
        else:
            # Just use the original coordinates
            new_coordinates = atcoords
            # Compute the unit vectors of the cubic grid's coordinate system
            axes = np.diag([spacing, spacing, spacing])

        # maximum and minimum value of x, y and z coordinates/grid.
        max_coordinate = np.amax(new_coordinates, axis=0)
        min_coordinate = np.amin(new_coordinates, axis=0)
        # Compute the required number of points along x, y, and z axis
        shape = (max_coordinate - min_coordinate + 2.0 * extension) / spacing
        # Add one to include the upper-bound as well.
        shape = np.ceil(shape)
        shape = np.array(shape, int)
        # Compute origin by taking the center of mass then subtracting the half of the number
        #    of points in the direction of the axes.
        origin = com - np.dot((0.5 * shape), axes)
        return cls(origin, axes, shape)

    def _calculate_volume(self, shape):
        r"""Return the volume of the Uniform Grid."""
        # Shape needs to be an argument, because I need to calculate the weights before
        #       initializing the _HyperRectangleGrid (where shape is set there).
        # Three-Dims: Volume of a parallelepiped spanned by a, b, c is  | (a x b) dot c|.
        if len(shape) == 3:
            volume = np.dot(
                np.cross(shape[0] * self.axes[0], shape[1] * self.axes[1]),
                shape[2] * self.axes[2],
            )
        else:
            # Two-Dims: Volume of a parallelogram is the absolute value of the determinant |a x b|
            volume = np.linalg.det(
                np.array([shape[0] * self.axes[0], shape[1] * self.axes[1]])
            )
        return np.abs(volume)

    def _choose_weight_scheme(self, shape):
        # Choose different weighting schemes.
        volume = self._calculate_volume(shape)
        numpnt = 1.0 * np.prod(shape)
        weights = np.full(np.prod(shape), volume / numpnt)
        return weights

    @property
    def points(self):
        """Return cubic grid points."""
        return self._points

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
        value = np.sum(self._weights * arr)
        return value
