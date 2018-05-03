r"""

"""
from fitting.radial_grid.radial_grid import RadialGrid
from numbers import Real, Integral
import numpy as np

__all__ = ["ClenshawGrid"]


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
        super(ClenshawGrid, self).__init__(g=grid)
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
