import numpy as np
import copy

class Grid_1D():
    max_num_of_pts_of_all_1D_grids = 0

    def __init__(self, type_of_grid, num_of_points, effort):
        assert type(num_of_points) is int, "num_of_points is not an integer: %r" % num_of_points
        assert type(type_of_grid) is str, "type_of_grid is not an string: %r" % type_of_grid
        assert type(effort) is int, "effort is not an integer: %r" % effort
        assert num_of_points > 0, "num_of_points should be positive: %r" % num_of_points
        assert effort  > 0, "effort should be positive: %r" % effort
        assert type_of_grid == "CC" or type_of_grid == "RR"

        self.type_of_grid = type_of_grid
        self.num_of_points = num_of_points
        self.effort = effort

        self.grid_1D = np.empty(self.num_of_points)
        self.weights_1D = np.empty(self.num_of_points)
        if self.type_of_grid == "CC":
            self.grid_1D = self.generate_clenshaw_curtis_1D_grid(num_of_points)
        elif self.type_of_grid == "RR":
            self.grid_1D = self.generate_rectangle_1D_grid(num_of_points)

        #Update maximum number of points for all grids
        if num_of_points > Grid_1D.max_num_of_pts_of_all_1D_grids:
            Grid_1D.max_num_of_pts_of_all_1D_grids = num_of_points

    def generate_clenshaw_curtis_1D_grid(self, num_of_points):
        for i in range(1, num_of_points + 1):
            self.grid_1D[i] = np.cos(i * np.pi / (num_of_points + 1))

            #TODO: FLOOR OR CEIL?
            coeff = []
            for x in range(0, int(np.num_of_points / 2.0) + 1):
                coeff.append(-4.0 * np.cos(2.0 * x * i * np.pi / (num_of_points + 1)) / (4.0 * x**2 - 1))

            self.weights_1D[i] = (((sum(coeff) - coeff[0]) / 2.0) - (coeff[-1] / 2.0)) / (num_of_points + 1)

        self.weights_1D = self.weights_1D / 2.0
        self.grid_1D = (1 + self.grid_1D) / 2.0


    def generate_rectangle_1D_grid(self, num_of_points):
        pass

    @staticmethod
    def get_clenshaw_curtis_1D_grid(delayed_sequence_number, effort):
        num_of_points = 0
        if delayed_sequence_number == 1:
            num_of_points = 1
        else:
            num_of_points = 2**(delayed_sequence_number - 1) - 1
        type_of_grid = "CC"

        return Grid_1D(type_of_grid, num_of_points, effort)

    @staticmethod
    def get_rectangle_1D_grid(delayed_sequence_number, effort):
        assert type(delayed_sequence_number) is int, "delayed_sequence_number is not an Integer: %r" % delayed_sequence_number

        type_of_grid = "RR"
        num_of_points = 2**(delayed_sequence_number) - 1

        return Grid_1D(type_of_grid, num_of_points, effort)



