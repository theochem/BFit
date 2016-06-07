import numpy as np
import copy
from fitting.grid_generation.grid_1D import Grid_1D
from fitting.grid_generation.grid_1D_generator import Grid_1D_Generator


class Grid_1D_Difference():
    def __init__(self, all_grids_obj):
        assert isinstance(all_grids_obj, Grid_1D_Generator), \
            "all_grids_obj is not of Class Grid_1D_Generator: instead type of all_grids_obj is %r" % type(all_grids_obj)

        self.all_grids_obj = all_grids_obj

        #Confirm Deletion
        #self.indexes_for_1D_difference = None
        #self.weights_for_1D_difference = None

        self.all_indexes_for_1D_difference = np.empty(shape=(self.all_grids_obj.effort,), dtype=object)
        self.all_weights_for_1D_difference = np.empty(shape=(self.all_grids_obj.effort,), dtype=object)


    def generate_difference_for_weights(self, index_of_grid):
        grid_i_obj = self.all_grids_obj.all_1D_grid_objs[index_of_grid]
        grid_i = grid_i_obj.grid_1D
        weight_i = grid_i_obj.weights_1D

        grid_i_min_1_obj = self.all_grids_obj.all_1D_grid_objs[index_of_grid]
        grid_i_min_1 = grid_i_min_1_obj.grid_1D
        weight_i_min_1 = grid_i_min_1_obj.weights_1D

        weights_for_1D_difference = np.zeros(grid_i_obj.num_of_points)

        for index_i in range(0, grid_i_obj.num_of_points):
            found = False
            for index_i_min_1 in range(0, grid_i_min_1_obj.num_of_points):
                if grid_i[index_i] == grid_i_min_1[index_i_min_1]:
                    weights_for_1D_difference[index_i] = weight_i[index_i] - weight_i_min_1[index_i_min_1]
                    found = True
                    break

            if not found:
                weights_for_1D_difference[index_i] = weight_i[index_i]

        return (weights_for_1D_difference)

    def generate_different_for_indexes(self, index_of_grid):
        grid_i_obj = self.all_grids_obj.all_1D_grid_objs[index_of_grid]
        grid_i = grid_i_obj.grid_1D

        indexes_for_1D_difference = np.zeros(grid_i_obj.num_of_points)

        for index in range(0, grid_i_obj.num_of_points):
            for index2 in range(0, self.all_grids_obj.max_num_of_pts_of_all_1D_grids):
                if self.all_grids_obj.labels_of_all_1D_grids[index2] == grid_i[index]:
                    indexes_for_1D_difference[index] = index2
                    break
        return (indexes_for_1D_difference)


    def generate_all_differences(self):
        ############################# Weights #############################################
        self.all_weights_for_1D_difference[0] = copy.deepcopy(self.all_grids_obj.all_1D_grid_objs[0].weights_1D)

        for x in range(1, self.all_grids_obj.effort):
            self.all_weights_for_1D_difference[x] = self.generate_difference_for_weights(x)

        ############################# Indexes #############################################
        for x in range(0, self.all_grids_obj.effort):
            self.all_indexes_for_1D_difference[x] = self.generate_different_for_indexes(x)