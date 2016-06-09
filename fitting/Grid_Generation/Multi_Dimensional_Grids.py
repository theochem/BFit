import numpy as np
import copy
from fitting.grid_generation.BST import BST, Node
from fitting.grid_generation.grid_1D_generator import Grid_1D_Generator
from fitting.grid_generation.grid_1D_difference import Grid_1D_Difference


class Multi_Dimensional_Grids():
    def __init__(self, all_grids_obj, dimension, T_GK=0, nsym=(0,0)):
        assert type(dimension) is int, "dimension is not an Integer: %r" % dimension
        assert isinstance(all_grids_obj, Grid_1D_Generator), \
            "all_grids_obj is not of Class Grid_1D_Generator: %r" % all_grids_obj

        self.dimension = dimension
        self.nsym = nsym
        self.T_GK = T_GK
        self.accepted_indexes = None
        self.grid_diff_obj = Grid_1D_Difference(all_grids_obj)
        self.grid_diff_obj.generate_all_differences()
        self.number_of_grid_points = None


    def is_index_acceptable(self, index_set):
        sum_index = np.sum(index_set)
        max_effort = self.grid_diff_obj.all_grids_obj.effort

        if self.T_GK < 1:
            test_bound = self.dimension + max_effort - 1 - self.T_GK * max_effort
            test_val = sum_index - self.T_GK * np.amax(index_set)
        else:
            test_bound = self.dimension + max_effort - 1 - (np.log(4**max_effort + 4 * self.dimension - 4) /\
                5 * np.log(2.))
            test_val = sum_index - (np.log(np.sum(4**index_set)) / (5 * np.log(2.)))

        if test_val <= test_bound:
            return True
        return False


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
        return set_of_all_possible_indices



    def smolyak_generation(self):
        index_set = np.array([0] + [1 for x in range(1, self.dimension)])
        number_of_points_in_smolyak = 0
        binary_search_tree = None
        counter = 0

        while counter <= self.dimension - 1:
            old_index = copy.deepcopy(index_set)
            index_set[counter] += 1

            if counter > 0:
                if self.nsym[0] == 0:
                    index_set[0:counter] = 1

                elif counter <= np.abs(self.nsym[0]):
                    index_set[0:counter]= index_set[counter]
                else:
                    if counter > abs(self.nsym[0]):
                        index_set[np.abs(self.nsym[0])+1:  counter] = index_set[counter]

            if self.is_index_acceptable(index_set):
                new_pts = 0
                for dim in range(0, self.dimension):
                    new_pts += self.grid_diff_obj.all_grids_obj.all_1D_grid_objs[index_set[dim] - 1].num_of_points
                number_of_points_in_smolyak += new_pts
                print(new_pts)
                if new_pts > 0:
                    binary_search_tree = self.make_point(binary_search_tree)
                counter = 0

            else:
                index_set = copy.deepcopy(old_index)
                counter += 1
        return self.make_array(binary_search_tree)

    def make_point(self, binary_tree):
        number_of_permutations = self.number_of_permutations_of_index_set(np.arange(1, self.grid_diff_obj.all_grids_obj.effort + 1))
        index_set = np.array([0] + [1 for x in range(1, self.dimension)])
        print("INDEX_SET", index_set)

        counter = 0
        while counter <= self.dimension - 1:
            old_index = copy.deepcopy(index_set)
            index_set[counter] += 1

            if counter > 0:
                index_set[0: counter] = 1

            if index_set[counter] <= self.grid_diff_obj.all_grids_obj.all_1D_grid_objs[counter].num_of_points:

                weight = 1.0
                index_of_points = []

                for i in range(0, self.dimension):
                    print(self.grid_diff_obj.all_weights_for_1D_difference[self.grid_diff_obj.all_grids_obj.all_1D_grid_objs[counter].effort - 1])
                    weight *= self.grid_diff_obj.all_weights_for_1D_difference[self.grid_diff_obj.all_grids_obj.all_1D_grid_objs[counter].effort - 1][index_set[i] - 1]
                    index_of_points.append(self.grid_diff_obj.all_indexes_for_1D_difference[self.grid_diff_obj.all_grids_obj.all_1D_grid_objs[counter].effort - 1] \
                                           [index_set[i] - 1])
                    print("i", i, self.dimension)
                weight *= number_of_permutations

                index_of_points, rejection = self.sort(index_of_points)

                if not rejection:
                    if binary_tree == None:
                        binary_tree = BST(index_of_points, weight)
                    else:
                        binary_tree.put_grid_point(index_of_points, weight)

                counter = 0

            else:
                index_set = np.copy(old_index)
                counter += 1


    def make_array(self, binary_search_tree):
        assert isinstance(binary_search_tree, BST), "binary_search_tree is not of type BST: %r" % type(binary_search_tree)
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
        return grid_point_array


    def sort(self, indices_of_grid_points):
        print(indices_of_grid_points)
        if self.nsym[0] != 0:
            while True:
                exchange = False

                for i in range(0, np.abs(self.nsym[0]) - 1):
                    if indices_of_grid_points[i] < indices_of_grid_points[i + 1]:
                        indices_of_grid_points[i], indices_of_grid_points[i + 1] = \
                        indices_of_grid_points[i + 1], indices_of_grid_points[i]
                        exchange = True
                if not exchange:
                    break
            while True:
                exchange = False
                for i in range(np.abs(self.nsym[0]) + 1, self.dimension):
                    print(i)
                    if indices_of_grid_points[i] < indices_of_grid_points[i + 1]:
                          indices_of_grid_points[i], indices_of_grid_points[i + 1] = \
                          indices_of_grid_points[i + 1], indices_of_grid_points[i]
                          exchange = True
                    if not exchange:
                        break
        if self.nsym[1] == 1:
            if Node.comparing_two_lists(indices_of_grid_points[0:(self.dimension / 2) + 1] ,
                                        indices_of_grid_points[(self.dimension / 2) + 1, self.dimension], "<"):
                indices_of_grid_points[0:(self.dimension / 2) + 1], indices_of_grid_points[(self.dimension / 2) + 1, self.dimension] = \
                indices_of_grid_points[(self.dimension / 2) + 1, self.dimension], indices_of_grid_points[0:(self.dimension / 2) + 1]
        reject = False
        if self.nsym[0] < 0:
            for i in range(0, np.abs(self.nsym[0])):
                if indices_of_grid_points[i] == indices_of_grid_points[i + 1]:
                    reject = True
                    break

            for i in range(np.abs(self.nsym[0]) + 1, self.dimension):
                if indices_of_grid_points[i] == indices_of_grid_points[i + 1]:
                    reject = True
                    break
        return  copy.deepcopy(indices_of_grid_points), reject


class IncorrectNumberOfPointsError(Exception):
    def __init__(self, expression, message):
        self.message = message
        self.expression = expression

if __name__ == "__main__":
    effort = 5
    delayed_sync = np.arange(1, effort + 1)
    all_grid_1D_obj = Grid_1D_Generator("CC", effort, delayed_sync)

    multi_dim_grid_obj = Multi_Dimensional_Grids(all_grid_1D_obj, 3)
    print(multi_dim_grid_obj.smolyak_generation())