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
        assert len(index_set) == self.dimension

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
        assert  len(list_of_indexes) == self.dimension

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
        assert len(index_set) == self.dimension
        number_of_points_in_smolyak = 0
        binary_search_tree = None
        counter = 0
        numb_of_pts= 0
        while counter <= self.dimension - 1:
            old_index = copy.deepcopy(index_set)
            index_set[counter] += 1

            if counter > 0:
                if self.nsym[0] == 0:
                    index_set[0:counter] = 1
                elif counter <= np.abs(self.nsym[0]):
                    index_set[0:counter]= index_set[counter]
                else:
                    index_set[0:np.abs(self.nsum[0])] = 1
                    if counter > np.abs(self.nsym[0]):
                        index_set[np.abs(self.nsym[0])+1:  counter] = index_set[counter]

            if self.is_index_acceptable(index_set):
                new_pts = 1
                for dim in range(0, self.dimension):
                    new_pts *= self.grid_diff_obj.all_grids_obj.all_1D_grid_objs[index_set[dim] - 1].num_of_points
                number_of_points_in_smolyak += new_pts

                if new_pts > 0:
                    binary_search_tree, number_of_pts = self.make_point(index_set, binary_search_tree)
                    numb_of_pts += number_of_pts
                counter = 0

            else:
                index_set = copy.deepcopy(old_index)
                counter += 1
        array_pts, weight_pts =  self.make_array(binary_search_tree, numb_of_pts)
        return array_pts, weight_pts , number_of_points_in_smolyak

    def make_point(self, grid_effort, binary_tree):
        assert len(grid_effort) == self.dimension
        assert binary_tree is None or isinstance(binary_tree, BST)

        number_of_permutations = self.number_of_permutations_of_index_set(grid_effort)
        index_set = np.array([0] + [1 for x in range(1, self.dimension)])
        assert len(index_set) == self.dimension

        counter = 0
        number_of_pts = 0
        while counter <= self.dimension - 1:
            old_index = copy.deepcopy(index_set)
            index_set[counter] += 1

            if counter > 0:
                index_set[0: counter] = 1

            if index_set[counter] <= self.grid_diff_obj.all_grids_obj.all_1D_grid_objs[grid_effort[counter] - 1].num_of_points:
                weight = 1.0
                index_of_points = []

                for i in range(0, self.dimension):
                    weight *= self.grid_diff_obj.all_weights_for_1D_difference[grid_effort[i] - 1][index_set[i] - 1]
                    index_of_points.append(self.grid_diff_obj.all_indexes_for_1D_difference[grid_effort[i] - 1][index_set[i] - 1])

                weight *= number_of_permutations

                assert len(index_of_points) == self.dimension

                index_of_points, rejection = self.sort(index_of_points)

                if not rejection:
                    newpt = True
                    if binary_tree == None:
                        binary_tree = BST(index_of_points, weight)
                    else:
                        newpt = binary_tree.put_grid_point(index_of_points, weight)
                    if newpt:
                        number_of_pts += 1
                counter = 0

            else:
                index_set = np.copy(old_index)
                counter += 1
        return binary_tree, number_of_pts

    def make_array(self, binary_search_tree, numb_of_pts):
        assert isinstance(binary_search_tree, BST), "binary_search_tree is not of type BST: %r" % type(binary_search_tree)
        count_number_of_grid_points = 0

        grid_point_list = [[] for x in range(0,self.dimension)]
        weight_list = []

        while binary_search_tree.root != None:
            count_number_of_grid_points += 1
            key, value = binary_search_tree.remove_grid_pt(binary_search_tree.root)
            #deleted_node = binary_search_tree.deleteMin()
            #assert isinstance(deleted_node, Node), "the minimum node delete is not a Node"
            weight_list.append(value)

            for i in range(0, self.dimension):
                #grid_point_list[i].append(self.grid_diff_obj.all_grids_obj.labels_of_all_1D_grids[deleted_node.key[i]])
                grid_point_list[i].append(self.grid_diff_obj.all_grids_obj.labels_of_all_1D_grids[key[i]])

        #if (count_number_of_grid_points != self.number_of_grid_points):
        #    raise IncorrectNumberOfPointsError("number of grid points generated != expected number",
        #                                       "Expected number of grid points does not match for make_array func")
        print("Number of grid points inside the BST is", count_number_of_grid_points, "Should be ", numb_of_pts)
        return  np.array(grid_point_list), np.array(weight_list)


    def sort(self, indices_of_grid_points):
        assert len(indices_of_grid_points) == self.dimension

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
                for i in range(np.abs(self.nsym[0]) + 1, self.dimension - 1):
                    if indices_of_grid_points[i] < indices_of_grid_points[i + 1]:
                          indices_of_grid_points[i], indices_of_grid_points[i + 1] = \
                          indices_of_grid_points[i + 1], indices_of_grid_points[i]
                          exchange = True
                    if not exchange:
                        break

        if self.nsym[1] == 1:
            if Node.comparing_two_lists(indices_of_grid_points[0:(self.dimension / 2) + 1] ,
                                        indices_of_grid_points[(self.dimension / 2) + 1:], "<"):
                indices_of_grid_points[0:(self.dimension / 2) + 1], indices_of_grid_points[(self.dimension / 2) + 1:] = \
                indices_of_grid_points[(self.dimension / 2) + 1:], indices_of_grid_points[0:(self.dimension / 2) + 1]

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
    effort = 10
    delayed_sync = []
    for x in np.arange(1, effort + 1):
        if x > 3:
            for y in range(0, int(2**(x - 3))):
                delayed_sync.append(x)
        else:
            delayed_sync.append(x)
    delayed_sync = np.arange(1, effort + 1)
    all_grid_1D_obj = Grid_1D_Generator("RR", effort, delayed_sync)

    multi_dim_grid_obj = Multi_Dimensional_Grids(all_grid_1D_obj, 2)
    array_pts, weight_pts, number_of_expected_pts = multi_dim_grid_obj.smolyak_generation()
    X = array_pts[0][1:]
    Y = array_pts[1][1:]

    import matplotlib.pyplot as plt
    plt.title("Rectangular Grid-Grid Smolyak Grid, Effort = 11")
    plt.text(0., -.12, "Delayed Sequence is [1,2, 3, 4, ..., effort]")
    plt.plot(X, Y, "ro")
    plt.xlabel("X1")
    plt.ylabel("Y1")
    plt.grid()
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    """
    Z = array_pts[2][1:]
    cartesian_prod = np.array([[x0, y0, z0] for x0 in X for y0 in Y for z0 in Z])#np.dstack(np.meshgrid(X, Y, Z)).reshape(-1, 3)
    print(cartesian_prod)
    X =cartesian_prod[:,0]
    Y = cartesian_prod[:,1]
    Z = cartesian_prod[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c="r", marker="o")
    plt.show()"""