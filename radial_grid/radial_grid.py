r"""
    Contains the radial _grid class used to define the domain of the least_squares models and
    to provide a function to integrate them. #TODO: Figure out What kind of Integration

    The properties we want for this _grid is to have
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
from numbers import Real, Integral

__all__ = ["RadialGrid", "ClenshawGrid", "HortonGrid"]


class RadialGrid(object):
    def __init__(self, grid):
        if not isinstance(grid, np.ndarray):
            raise TypeError("Grid should be a numpy array.")
        if grid.ndim != 1:
            raise ValueError("Grid should be one dimensional.")
        self._radii = np.ravel(grid)

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

    def integrate(self, *args):
        #TODO: I NEED TO TEST THIS. AND THE INTEGRATE SPHERICALLY AS I HTINK I MADE AN ERROR.
        total_arr = np.ma.asarray(np.ones(len(args[0])))
        for arr in args:
            total_arr *= arr
        return np.trapz(y=total_arr, x=self.radii)


class ClenshawGrid(RadialGrid):
    """
    Constructs a clenshaw-curtis _grid and provides a function to integrate_spher over
    the entire space. #TODO:DifferentKind of Integration

    This _grid is used to concentrate more points near the origin/nucleus and
    add sparse points further away.
    """
    def __init__(self, atomic_number, numb_core_pts, numb_diffuse_pts, extra_list=[],
                 filled=False):
        """
        Parameters
        ----------
        atomic_number : int
                        Atomic number of the atom that is being modeled. This is used
                        to generate the clenshaw-curtis _grid.
        numb_core_pts : int
                        The number of points that should be concentrated near the
                        origin/nucleus.
        numb_diffuse_pts : int
                           The number of points that are far from the origin/nucleus.
        extra_list : list
                     Extra points that one wants to add on the _grid.
        filled : bool
                 Used for integration #TODO:Clarrify This

        Raises
        ------
        TypeError
            If an argument of an invalid type is used
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
        grid = self.grid_points(numb_core_pts, numb_diffuse_pts, extra_list)
        super(RadialGrid, self).__init__(grid)
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
        interval = np.arange(0, numb_pts)
        factor = 1 / (2 * self._atomic_numb)
        core_grid = factor * (1 - np.cos(np.pi * interval / (2 * numb_pts)))
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
        interval = np.arange(0, numb_pts)
        diffuse_grid = 25. * (1. - np.cos(np.pi * interval / (2. * numb_pts)))
        return diffuse_grid

    def grid_points(self, numb_core_pts, numb_diffuse_pts, extra_list=()):
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

    @staticmethod
    def uniform_grid(number_of_points):
        return np.arange(100, step=1/number_of_points)


class HortonGrid(RadialGrid):
    def __init__(self, smallest_pt, largest_pt, numb_pts, filled=False):
        if not isinstance(smallest_pt, Real):
            raise TypeError("smallest point should be a number.")
        if not isinstance(largest_pt, Real):
            raise TypeError("largest point should be a number.")
        if not isinstance(numb_pts, Integral):
            raise TypeError("number of points should be an integer.")
        if numb_pts <= 0.:
            raise ValueError("number of points should be positive.")
        import horton
        rtf = horton.ExpRTransform(smallest_pt, largest_pt, numb_pts)
        self._radial_grid = horton.RadialGrid(rtf)
        super(RadialGrid, self).__init__(self.radial_grid.radii.copy())
        self._filled = filled

    @property
    def radii(self):
        return self._radii

    @property
    def radial_grid(self):
        return self._radial_grid

    def integrate(self, *args):
        total_arr = np.ma.asarray(np.ones(len(args[0])))
        for arr in args:
            total_arr *= arr
        #if self._filled:
        #    total_arr = np.ma.filled(total_arr, 0.)
        return self.radial_grid.integrate(total_arr)
