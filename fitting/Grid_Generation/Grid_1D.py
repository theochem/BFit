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


        if self.type_of_grid == "CC":
            self.grid_1D, self.weights_1D = self.generate_clenshaw_curtis_1D_grid(num_of_points)
        elif self.type_of_grid == "RR":
            self.grid_1D, self.weights_1D = self.generate_rectangle_1D_grid(num_of_points)

        #Update maximum number of points for all grids
        if num_of_points > Grid_1D.max_num_of_pts_of_all_1D_grids:
            Grid_1D.max_num_of_pts_of_all_1D_grids = num_of_points

    def generate_clenshaw_curtis_1D_grid(self, num_of_points):
        grid_1D = np.empty(num_of_points)
        weights_1D = np.empty(num_of_points)

        coefficients = np.zeros(num_of_points)
        if num_of_points == 1:
            grid_1D[0] = 0.5
            weights_1D[0] = 1.0

        else:
            for i in range(0, num_of_points):
                grid_1D[i] = np.cos((i+1) * np.pi / (num_of_points + 1))

                #TODO: FLOOR OR CEIL?
                coefficients = np.zeros(int((num_of_points + 1) / 2.0) + 1)
                assert len(coefficients) == int((num_of_points + 1) / 2.0) + 1
                for x in range(0, int((num_of_points + 1) / 2.0) + 1):
                    coefficients[x] = -4.0 * np.cos(2.0 * x * (i+1) * np.pi / (num_of_points + 1)) / (4.0 * x**2 - 1)
                weights_1D[i] = (np.sum(coefficients) - (coefficients[0] / 2.0) - \
                                 (coefficients[int((num_of_points + 1) / 2.0) - 1] / 2.0) )/ (num_of_points + 1)


            weights_1D = weights_1D / 2.0
            grid_1D = (1 + grid_1D) / 2.0

        return(grid_1D, weights_1D)



    def generate_rectangle_1D_grid(self, num_of_points):
        grid_1D = np.empty(num_of_points)
        weights_1D = np.empty(num_of_points)
        n_plus_one = num_of_points + 1
        for x in range(1, num_of_points + 1):
            weights_1D[x - 1] = 1.0 / n_plus_one
            grid_1D[x - 1] = x / n_plus_one
        return grid_1D, weights_1D


    @staticmethod
    def get_clenshaw_curtis_1D_grid_object(delayed_sequence_number, effort):
        num_of_points = Grid_1D.get_number_of_points_for_CC(delayed_sequence_number)
        type_of_grid = "CC"
        return Grid_1D(type_of_grid, num_of_points, effort)


    @staticmethod
    def get_number_of_points_for_CC(delayed_sequence_number):
        if delayed_sequence_number == 1:
            num_of_points = 1
        else:
            #TODO: IS THIS RIGHT?
            num_of_points = int(2**(delayed_sequence_number - 1) - 1)
        return num_of_points

    @staticmethod
    def get_rectangle_1D_grid(delayed_sequence_number, effort):
        #assert type(delayed_sequence_number) is int, "delayed_sequence_number is not an Integer: %r" % delayed_sequence_number

        type_of_grid = "RR"
        num_of_points = int(2**(delayed_sequence_number)) - 1

        return Grid_1D(type_of_grid, num_of_points, effort)



