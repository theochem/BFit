# -*- coding: utf-8 -*-
# An basis-set curve-fitting optimization package.
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
    Contains the radial_grid class used to define the domain of the least_squares models and
    to provide a function to integrate them.

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

import numpy as np

__all__ = ["RadialGrid"]


class RadialGrid(object):

    def __init__(self, g):
        if not isinstance(g, np.ndarray):
            raise TypeError("Grid should be a numpy array.")
        if g.ndim != 1:
            raise ValueError("Grid should be one dimensional.")
        self._radii = np.ravel(g)

    @property
    def radii(self):
        return self._radii

    def integrate_spher(self, *args, filled=False):
        r"""
        Integrates a _grid on the radii points in a spherical
        format.
        ..math::
            \int 4 \pi r^2 f(r)dr, \\
            where f(r) is what is being integrated.

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
