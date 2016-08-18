import scipy
import numpy as np;

class Radial_Grid():
    def __init__(self, atomic_number, num_of_core_points, num_of_diffuse_points, extra_list):
        assert type(atomic_number) is int, "Atomic number has to be an integer"
        self.atomic_number = atomic_number
        self.radii = self.grid_points(num_of_core_points, num_of_diffuse_points, extra_list=extra_list)

    def core_points(self, num_points):
        assert type(num_points) is int, "Grid points is not an integer"


        interval = np.arange(0, num_points)
        core_grid = 1/(2 * self.atomic_number) * (1 - np.cos(np.pi * interval / (2 * num_points)))
        assert len(core_grid) == num_points

        return(core_grid)

    def diffuse_points(self, num_points):
        assert type(num_points) is int, "Number of points has to be an integer"

        interval = np.arange(0, num_points)
        diffuse_grid = 25 * (1 - np.cos(np.pi * interval / (2 * num_points)))
        assert len(diffuse_grid) == num_points

        return(diffuse_grid)

    def grid_points(self, num_of_core_points, num_of_diffuse_points, extra_list=[]):
        """
        This grid is based on the Clenshaw-Curtis idea
        :param num_of_core_points: uses points from core_points function
        :param num_of_diffuse_points: creates points from diffuse_points function
        :param extra_list: this is optional list that can be added
        :return:
        """

        core_points = self.core_points(num_of_core_points)
        #[1:] is used to remove the extra zero in diffuse grid
        #because there exists an zero already in core_points
        diffuse_points = self.diffuse_points(num_of_diffuse_points)[1:]

        grid_points = np.concatenate((core_points, diffuse_points))
        grid_points = np.concatenate((grid_points, extra_list))
        sorted_grid_points = np.sort(grid_points)

        return(sorted_grid_points)

    def integrate(self, arr):
        return np.trapz(y=arr * 4 * np.pi * np.power(self.radii, 2.), x=self.radii)

class Horton_Grid():
    pass