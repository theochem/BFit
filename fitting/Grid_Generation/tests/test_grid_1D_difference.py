from nose.tools import assert_equals, assert_not_equal
import unittest
from fitting.grid_generation.grid_1D_generator import Grid_1D_Generator
from fitting.grid_generation.grid_1D_difference import Grid_1D_Difference
from nose.tools import with_setup
import numpy as np

class Test_Grid_1D_diff(unittest.TestCase):
    def setUp(self):
        self.type_of_all_grids = "CC"
        self.effort = 5
        self.delayed_sequence = np.arange(1, self.effort + 1)
        self.grid_1D_generator = Grid_1D_Generator(self.type_of_all_grids, self.effort, self.delayed_sequence)

        self.grid_1D_difference = Grid_1D_Difference(self.grid_1D_generator)
        self.grid_1D_difference.generate_all_differences()

    def teardown(self):
        print ("TestUM:teardown() after each test method")

    @classmethod
    def setup_class(cls):
        print ("setup_class() before any methods in this class")

    @classmethod
    def teardown_class(cls):
        print ("teardown_class() after any methods in this class")

    def Test_What(self):
        assert 1 == 1
        import matplotlib.pyplot as plt
        list_of_colours = ["b", "g", "r", "c", "m", "y", "k", "w"]
        for x in range(0, self.effort):
            print(self.grid_1D_difference.all_indexes_for_1D_difference[x])
            plt.plot(self.grid_1D_difference.all_indexes_for_1D_difference[x], list_of_colours[x]+"o", label="diff="+str(x)+"&"+str(x-1))
            plt.plot(self.grid_1D_generator.all_1D_grid_objs[x].grid_1D, list_of_colours[x]+"s", label="effort="+str(x))
        plt.legend()
        plt.show()
