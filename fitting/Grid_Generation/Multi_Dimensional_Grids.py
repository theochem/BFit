import numpy as np
import copy
from fitting.grid_generation.BST import BST, Node
from fitting.grid_generation.grid_1D_generator import Grid_1D_Generator
from fitting.grid_generation.grid_1D_difference import Grid_1D_Difference


class Multi_Dimensional_Grids():
    def __init__(self, all_grids_obj, dimension, T_GK=0, nsym=(2,1)):
        assert type(dimension) is int, "dimension is not an Integer: %r" % dimension
        assert isinstance(all_grids_obj, Grid_1D_Generator), \
            "all_grids_obj is not of Class Grid_1D_Generator: %r" % all_grids_obj

        self.dimension = dimension
        self.nsym = nsym
        self.T_GK = 0
        self.accepted_indexes = None
        self.grid_diff_obj = Grid_1D_Difference(all_grids_obj)
        self.grid_diff_obj.generate_all_differences()
        self.number_of_grid_points = None


    def is_index_acceptable(self, index_set):
        sum_index = np.sum(index_set)
        if self.T_GK == 0:
            return True if sum_index <= self.grid_diff_obj.all_grids_obj.effort + self.dimension - 1 else False


    def number_of_permutations_of_index_set(self, list_of_indexes):
        number_of_permutations = 0
        if self.nsym[0] == 0:
            number_of_permutations = 1
        else:
            number_of_permutations =  np.math.factorial(np.abs(self.nsym[0])) * np.math.factorial(self.dimension - np.abs(self.nsym[0]))

            for x in range(0, np.abs(self.nsym[0])):
                count = 1
                for y in range(x+1, np.abs(self.nsym[0])):
                    if list_of_indexes[y] == list_of_indexes[x]:
                        count += 1

                number_of_permutations /= count

            for x in range(np.abs(self.nsym[0]), self.dimension):
                count = 1
                for y in range(x+1, self.dimension):
                    if list_of_indexes[x] == list_of_indexes[y]:
                        count += 1
                number_of_permutations /= count

        return number_of_permutations




    def odometer_algo(self):
        """
        WASTEEEE
        :return:
        """
        set_of_indexes = np.ones(self.dimension )
        set_of_all_possible_indices = [copy.deepcopy(set_of_indexes)]
        counter = 0
        num_of_pts = 1
        while counter <= self.dimension - 1:
            oldIndex = copy.deepcopy(set_of_indexes)
            set_of_indexes[counter] += 1
            for i in range(0, counter - 1):
                set_of_indexes[i] = set_of_indexes[counter]

            if self.is_index_acceptable(set_of_indexes):
                num_of_pts += 1
                set_of_all_possible_indices.append(copy.deepcopy(set_of_indexes))
                counter = 1
            else:
                set_of_indexes = copy.deepcopy(oldIndex)
                counter += 1
        return (set_of_all_possible_indices)

    def smolyak_generation(self):
        pass


    def make_point(self):
        number_of_permutations = self.number_of_permutations_of_index_set(np.arange(1, self.grid_diff_obj.all_grids_obj.effort + 1))
        index_set = np.array([0] + [1 for x in range(1, self.dimension)])
        print(index_set)

        counter = 0
        while counter <= self.dimension - 1:
            old_index = copy.deepcopy(index_set)
            index_set[counter] += 1

            if counter > 1:
                index_set[1: counter] = 1

            if index_set[counter] <= self.grid_diff_obj.all_grids_obj.all_1D_grid_objs[counter].num_of_points:

                counter = 1





            else:
                index_set = np.copy(old_index)
                counter += 1

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