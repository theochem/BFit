from nose.tools import assert_equals, assert_not_equal
import unittest
from fitting.grid_generation.grid_1D_generator import Grid_1D_Generator
from nose.tools import with_setup
import numpy as np

class Test_Grid_1D_For_CC(unittest.TestCase):
    def setUp(self):
        self.type_of_all_grids = "CC"
        self.effort = 5
        self.delayed_sequence = np.arange(1, self.effort + 1)
        self.grid_1D_generator = Grid_1D_Generator(self.type_of_all_grids, self.effort, self.delayed_sequence)

    def teardown(self):
        print ("TestUM:teardown() after each test method")

    @classmethod
    def setup_class(cls):
        print ("setup_class() before any methods in this class")

    @classmethod
    def teardown_class(cls):
        print ("teardown_class() after any methods in this class")

    def test_one_point_grids(self):

        assert len(self.grid_1D_generator.all_1D_grid_objs[0].grid_1D) == 1
        assert len(self.grid_1D_generator.all_1D_grid_objs[0].grid_1D) == 1
        assert self.grid_1D_generator.all_1D_grid_objs[0].grid_1D[0] == 0.5
        assert self.grid_1D_generator.all_1D_grid_objs[0].weights_1D[0] == 1

        import matplotlib.pyplot as plt
        plt.plot(self.grid_1D_generator.all_1D_grid_objs[0].grid_1D, 'ro')
        plt.plot(self.grid_1D_generator.all_1D_grid_objs[1].grid_1D, 'bo')
        plt.plot(self.grid_1D_generator.all_1D_grid_objs[2].grid_1D, 'go')
        plt.plot(self.grid_1D_generator.all_1D_grid_objs[3].grid_1D, 'yo')
        plt.plot(self.grid_1D_generator.all_1D_grid_objs[4].grid_1D, 'go')
        plt.show()

    def test_number_of_points(self):
        assert len(self.grid_1D_generator.all_1D_grid_objs[2].grid_1D) == int(2**(self.delayed_sequence[2] - 1) - 1)
        assert len(self.grid_1D_generator.all_1D_grid_objs[3].grid_1D) == int(2**(self.delayed_sequence[3] - 1) - 1)
        assert len(self.grid_1D_generator.all_1D_grid_objs[4].grid_1D) == int(2**(self.delayed_sequence[4] - 1) - 1)

    def test_indexes(self):
        assert len(self.grid_1D_generator.labels_of_all_1D_grids) == len( self.grid_1D_generator.labels_of_all_1D_grids)
        for x in range(0, len(self.grid_1D_generator.labels_of_all_1D_grids)):
            assert (self.grid_1D_generator.all_1D_grid_objs[4].grid_1D[x]) == self.grid_1D_generator.labels_of_all_1D_grids[x]


if __name__ == "__main__":
    unittest.main()