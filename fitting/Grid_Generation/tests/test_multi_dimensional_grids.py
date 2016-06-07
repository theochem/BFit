from nose.tools import assert_equals, assert_not_equal
import unittest
from fitting.grid_generation.multi_dimensional_grids import Multi_Dimensional_Grids
from fitting.grid_generation.grid_1D_generator import Grid_1D_Generator
from nose.tools import with_setup
import numpy as np

class Test_Mult_Dim_Grid(unittest.TestCase):
    def setUp(self):
        self.type_of_all_grids = "CC"
        self.effort = 5
        self.delayed_sequence = np.arange(1, self.effort + 1)
        self.grid_1D_generator = Grid_1D_Generator(self.type_of_all_grids, self.effort, self.delayed_sequence)
        self.dimension = 3

        self.multi_dim_grid_obj = Multi_Dimensional_Grids(self.grid_1D_generator, self.dimension)

    def teardown(self):
        print ("TestUM:teardown() after each test method")

    @classmethod
    def setup_class(cls):
        print ("setup_class() before any methods in this class")

    @classmethod
    def teardown_class(cls):
        print ("teardown_class() after any methods in this class")


    def Test_Permutations(self):
        assert 1 == 1
        print(self.multi_dim_grid_obj.odometer_algo())
        print(self.multi_dim_grid_obj.number_of_permutations_of_index_set(np.array([1., 2., 3.])))
        print(self.multi_dim_grid_obj.make_point())