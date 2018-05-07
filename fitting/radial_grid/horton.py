r"""

"""

from fitting.radial_grid.general_grid import RadialGrid
from numbers import Real, Integral
import numpy as np

__all__ = ["HortonGrid"]


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
        super(HortonGrid, self).__init__(self.radial_grid.radii.copy())
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
        # if self._filled:
        #    total_arr = np.ma.filled(total_arr, 0.)
        return self.radial_grid.integrate(total_arr)
