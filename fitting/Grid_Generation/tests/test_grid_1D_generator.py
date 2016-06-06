from nose.tools import assert_equals, assert_not_equal
import unittest
from fitting.grid_generation.grid_1D_generator import Grid_1D_Generator
from nose.tools import with_setup


class Test_Grid_1D(unittest.TestCase):
    def setUp(self):
        self.type_of_all_grids = "CC"
        self.effort = 5
        self.grid_1D_generator = Grid_1D_Generator(self.type_of_all_grids, self.effort)

    def teardown(self):
        print ("TestUM:teardown() after each test method")

    @classmethod
    def setup_class(cls):
        print ("setup_class() before any methods in this class")

    @classmethod
    def teardown_class(cls):
        print ("teardown_class() after any methods in this class")

    def test_grids(self):
        import matplotlib.pyplot as plt
        plt.plot(self.grid_1D_generator.all_1D_grids[0])
        plt.plot(self.grid_1D_generator.all_1D_grids[1])
        plt.plot(self.grid_1D_generator.all_1D_grids[2])
        plt.plot(self.grid_1D_generator.all_1D_grids[3])
        plt.plot(self.grid_1D_generator.all_1D_grids[4])
        plt.show()



if __name__ == "__main__":
    unittest.main()