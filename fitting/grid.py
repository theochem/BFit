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
"""The Grid Module."""


import numpy as np

from numbers import Real, Integral


__all__ = ["BaseRadialGrid", "ClenshawRadialGrid", "CubicGrid"]


class BaseRadialGrid(object):
    """Base atom-centered radial grid."""

    def __init__(self, radii):
        """
        """
        if not isinstance(radii, np.ndarray) or radii.ndim != 1:
            raise TypeError("Argument radii should be one dimensional numpy array.")
        self._radii = np.ravel(radii)

    @property
    def radii(self):
        return self._radii

    def integrate_spher(self, filled=False, *args):
        r"""Compute spherical integration of given functions on radial grid points.

        ..math::
            \int 4 \pi r^2 f(r)dr

        where :math:`f(r)` is what is being integrated.

        Parameters
        ----------
        *args :
              Arguments of arrays to be multiplied together
              before integrating.

        filled : bool
                 If the arguments are masked array. Fills in the missing value
                 with zero.

        Returns
        -------
        float
            Integration value
        """
        total_arr = np.ma.asarray(np.ones(len(args[0])))
        for arr in args:
            total_arr *= arr
        if filled:
            total_arr = np.ma.filled(total_arr, 0.)
        integrand = total_arr * np.power(self.radii, 2.)
        return 4. * np.pi * np.trapz(y=integrand, x=self.radii)

    def integrate(self, *args):
        """
        """
        total_arr = np.ma.asarray(np.ones(len(args[0])))
        for arr in args:
            total_arr *= arr
        return np.trapz(y=total_arr, x=self.radii)


class UniformRadialGrid(BaseRadialGrid):
    """ """

    def __init__(self, number_points, max_radii=100.):
        """
        """
        grid = np.arange(max_radii, step=1./number_points)
        super(UniformRadialGrid, self).__init__(grid)


class ClenshawRadialGrid(BaseRadialGrid):
    """
    Constructs a clenshaw-curtis _grid and provides a function to integrate_spher over
    the entire space. #TODO:DifferentKind of Integration

    This _grid is used to concentrate more points near the origin/nucleus and
    add sparse points further away.

    The properties we want for this grid is to have
    points that are dense near the core region and another _grid that is
    spreads out the points.

    This is constructed as follows:

     ..math::
    The core points added is defined based on : \\
        r_p &= \frac{1}[2Z} (1 - cos(\frac{\pi p}{2N})) for p=0,1...N-1,
        where Z is the atomic number and N is the number of points. \\
    The diffuse points added is defined based on : \\
        r_p &= 25 (1 - cos(\frac{\pi p}{2N})) for p =0,1..N,
        where N is the number of points.
    Extra points are added to ensure better accuracy,
        r_p &=[50, 75, 100].

    """

    def __init__(self, atomic_number, numb_core_pts, numb_diffuse_pts, extra_list=[], filled=False):
        """
        Parameters
        ----------
        atomic_number : int
            Atomic number of the atom for which the grid is generated.
        numb_core_pts : int
            The number of points near the origin/nucleus.
        numb_diffuse_pts : int
            The number of points far from the origin/nucleus.
        extra_list : list
            Additional points to be added to the grid.
        filled : bool
            Used for integration #TODO:Clarrify This
        """
        if not isinstance(atomic_number, Integral):
            raise TypeError("atomic number should be an integer.")
        if atomic_number <= 0.:
            raise ValueError("atomic number should be positive.")
        if not isinstance(numb_core_pts, Integral):
            raise TypeError("Number of core points should be an integer.")
        if numb_core_pts <= 0.:
            raise ValueError("Number of core points should be positive.")
        if not isinstance(numb_diffuse_pts, Integral):
            raise TypeError("Number of diffuse points should be an integer.")
        if numb_diffuse_pts <= 0.:
            raise ValueError("Number of diffuse points should be positive.")
        if not (isinstance(extra_list, list) or isinstance(extra_list, tuple)):
            raise TypeError("Extra points to be added should be contained in a list.")

        self._atomic_numb = atomic_number
        grid = self._grid_points(numb_core_pts, numb_diffuse_pts, extra_list)
        super(ClenshawRadialGrid, self).__init__(grid)
        self._filled = filled

    @property
    def atomic_numb(self):
        return self._atomic_numb

    def _get_core_points(self, numb_pts):
        r"""
        Concentrates points on the radial _grid on [0, inf)
        near the origin for better accuracy.
        More specifically it is:
        ..math::
            r_p = \frac{1}[2Z} (1 - cos(\frac{\pi p}{2N})) for p=0,1...N-1,
        where Z is the atomic number and N is the number of points.

        Parameters
        ----------
        numb_pts : int
                  Number of core points on the _grid.

        Returns
        -------
        array
            Numpy array holding the core points on [0, inf)
        """
        assert type(numb_pts) is int, "Grid points is not an integer"
        factor = 1. / (2 * self._atomic_numb)
        core_grid = factor * (1 - np.cos(0.5 * np.pi * np.arange(0, numb_pts) / numb_pts))
        return core_grid

    def _get_diffuse_pts(self, numb_pts):
        r"""
        Get points concentrated away from the origin on [0, inf).

        More specifically it is:
        ..math::
            r_p = 25 (1 - cos(\frac{\pi p}{2N})) for p =0,1..N-1,
        where N is the number of points

        Parameters
        ----------
        numb_pts : int
            Number of diffuse points on the _grid.

        Returns
        -------
        array
            Numpy array holding the diffuse points on [0, inf)
        """
        assert type(numb_pts) is int, "Number of points has to be an integer"
        diffuse_grid = 25. * (1. - np.cos(0.5 * np.pi * np.arange(0, numb_pts) / numb_pts))
        return diffuse_grid

    def _grid_points(self, numb_core_pts, numb_diffuse_pts, extra_list=()):
        r"""
        Returns _grid points on the radial _grid, ie [0, inf),
        based on the clenshaw curtis _grid, where points are
        concentrated near the origin.

        Parameters
        ----------
        numb_core_pts : int
            Number of core points to add.
        numb_diffuse_pts : int
            Number of diffuse points to add.
        extra_list : list
            Add extra, specific points.

        Returns
        -------
        array
            Numpy array holding both core and diffuse points.
        """
        core_points = self._get_core_points(numb_core_pts)
        # [1:] is used to remove the extra zero in diffuse _grid
        # because there exists an zero already in core_points
        diffuse_points = self._get_diffuse_pts(numb_diffuse_pts)[1:]
        grid_points = np.concatenate((core_points, diffuse_points, extra_list))
        sorted_grid_points = np.sort(grid_points)
        return sorted_grid_points


class CubicGrid(object):
    """Cubic Grid Class."""

    def __init__(self, smallest_pt, largest_pt, step_size):
        """
        """
        if not isinstance(smallest_pt, Real):
            raise TypeError("Smallest point should be a number.")
        if not isinstance(largest_pt, Real):
            raise TypeError("Largest point should be a number.")
        if not isinstance(step_size, Real):
            raise TypeError("Step-size should be a number.")
        if not largest_pt > smallest_pt:
            raise ValueError("Largest point should be greater than smallest pointt.")
        if not step_size > 0.:
            raise ValueError("Step-size should be positive, non-zero.")
        self.step = step_size
        grid = CubicGrid.make_cubic_grid(smallest_pt, largest_pt, step_size)
        self._grid = grid

    @property
    def grid(self):
        return self._grid

    def __len__(self):
        return self._grid.shape[0]

    def integrate_spher(self, filled=False, *args):
        total_arr = np.ma.asarray(np.ones(len(args[0])))
        for arr in args:
            total_arr *= arr
        return self.step**3. * np.sum(total_arr)

    @staticmethod
    def make_cubic_grid(smallest_pt, largest_pt, step_size):
        grid = []
        grid_1d = np.arange(smallest_pt, largest_pt + step_size, step_size)
        for x in grid_1d:
            for y in grid_1d:
                for z in grid_1d:
                    grid.append([x, y, z])
        return np.array(grid)
