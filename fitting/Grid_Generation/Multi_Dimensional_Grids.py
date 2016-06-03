import numpy as np
import copy
from fitting.grid_generation.BST import BST, Node
from fitting.grid_generation.grid_1D_generator import Grid_1D_Generator
from fitting.grid_generation.grid_1D_difference import Grid_1D_Difference


class Multi_Dimensional_Grids():
    def __init__(self, all_grids_obj, dimension):
        assert type(dimension) is int, "dimension is not an Integer: %r" % dimension
        assert isinstance(all_grids_obj, Grid_1D_Generator), \
            "all_grids_obj is not of Class Grid_1D_Generator: %r" % all_grids_obj

        self.dimension = dimension
        self.accepted_indexes = None
        self.grid_diff_obj = Grid_1D_Difference(all_grids_obj)
        self.grid_diff_obj.generate_all_differences()
        self.number_of_grid_points = None


    def is_index_acceptable(self, index_set):
        sum_index = np.sum(index_set)
        return True if sum_index <= self.grid_diff_obj.all_grids_obj.effort + self.dimension - 1 else False




    def find_permutations_of_index_set(self):
        """
        WASTEEEE
        :return:
        """
        set_of_indexes = np.arange(1, self.dimension + 1)
        set_of_all_possible_indices = [set_of_indexes]
        counter = 1
        num_of_pts = 1

        while counter <= self.dimension:
            oldIndex = copy.deepcopy(set_of_indexes)
            set_of_indexes[counter] += 1
            for i in range(1, counter - 1):
                set_of_indexes[i] = set_of_indexes[counter]

            if self.is_index_acceptable(set_of_indexes):
                num_of_pts += 1
                set_of_all_possible_indices.append(set_of_indexes)
                counter = 1
            else:
                set_of_indexes = copy.deepcopy(oldIndex)
                counter += 1
        return (set_of_all_possible_indices)

    def smolyak_generation(self):
        pass

    def make_array(self, binary_search_tree):
        assert isinstance(binary_search_tree, BST), "binary_search_tree is not of type BST: %r" % range
        count_number_of_grid_points = 0
        grid_point_array = np.empty(shape=(self.dimension,), dtype=object)

        while binary_search_tree.root != None:
            count_number_of_grid_points += 1
            deleted_node = binary_search_tree.deleteMin()[1]
            assert isinstance(deleted_node, Node), "the minimum node delete is not a Node"
            for i in range(0, self.dimension):
                grid_point_array[i, count_number_of_grid_points]\
                    = self.grid_diff_obj.all_indexes_for_1D_difference[i, deleted_node.key]


        if (count_number_of_grid_points != self.number_of_grid_points):
            raise IncorrectNumberOfPointsError("number of grid points generated != expected number",
                                               "Expected number of grid points does not match for make_array func")

class IncorrectNumberOfPointsError(Exception):
    def __init__(self, expression, message):
        self.message = message
        self.expression = expression