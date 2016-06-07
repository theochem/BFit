import numpy as np
from fitting.grid_generation.grid_1D import Grid_1D
import copy

class Grid_1D_Generator():
    def __init__(self, type_of_all_grids, effort, delayed_sequence):
        assert type(type_of_all_grids) is str, "type_of_all_grids is not an string" % type_of_all_grids
        assert type(effort) is int, "effort is not an integer %r" % effort
        assert type_of_all_grids == "RR" or type_of_all_grids == "CC",\
            "type_of_all_grids is not specified/recognized: %r" % type_of_all_grids

        self.type_of_all_grids = type_of_all_grids
        self.effort = effort
        self.delayed_sequence = delayed_sequence

        self.all_1D_grid_objs = self.generate_all_1D_grids()

        self.labels_of_all_1D_grids = copy.deepcopy(self.all_1D_grid_objs[self.effort - 1].grid_1D)
        self.max_num_of_pts_of_all_1D_grids = Grid_1D.max_num_of_pts_of_all_1D_grids
        Grid_1D.max_num_of_pts_of_all_1D_grids = 0 #Set to 0 to use another generator again


    def generate_all_1D_grids(self):
        all_grids = np.empty(self.effort, dtype=object)
        if self.type_of_all_grids == "CC":
            for i in range(0, self.effort):
                all_grids[i] = Grid_1D.get_clenshaw_curtis_1D_grid_object(self.delayed_sequence[i], i+1)

        elif self.type_of_all_grids == "RR":
            for i in range(0, self.effort):
                all_grids[i] = Grid_1D.get_rectangle_1D_grid(self.delayed_sequence[i], i+1)
        return(all_grids)

    def generate_indexes_for_all_grids(self):
        indexes_for_1D_grids = []
        counter = 0
        for i in range(self.effort, 0, -1):
            grid_1D_obj = self.all_1D_grid_objs[i - 1]
            for point in grid_1D_obj.grid_1D:
                indexes_for_1D_grids.append(counter)
                counter += 1
        return(np.array(indexes_for_1D_grids))








class Grid_1D_Difference_Container():
    def __init__(self):
        pass

    def generate_all_1D_difference(self):
        pass



if __name__ == "__main__":
    pass