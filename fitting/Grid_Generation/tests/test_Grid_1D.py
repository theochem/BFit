from nose.tools import assert_equals, assert_not_equal
import unittest
from fitting.grid_generation.grid_1D import Grid_1D
from nose.tools import with_setup

class Test_Grid_1D_Creation(unittest.TestCase):
    def setUp(self):
       pass


    def teardown(self):
        print ("TestUM:teardown() after each test method")

    @classmethod
    def setup_class(cls):
        print ("setup_class() before any methods in this class")

    @classmethod
    def teardown_class(cls):
        pass


class TestClenshawGridGeneration(unittest.TestCase):
    def setUp(self):
        self.type_of_grid = "CC"
        self.num_of_points = 300
        self.effort = 5
        self.clenshaw_grid = Grid_1D(self.type_of_grid, self.num_of_points, self.effort)

    def teardown(self):
        print ("TestUM:teardown() after each test method")

    @classmethod
    def setup_class(cls):
        print ("setup_class() before any methods in this class")

    @classmethod
    def teardown_class(cls):
        pass

    def test_range_of_values_in_grid(self):
        assert (self.clenshaw_grid.grid_1D >= 0.0).all()
        assert (self.clenshaw_grid.grid_1D <= 1.0).all()

    def test_number_of_points(self):
        assert(len(self.clenshaw_grid.grid_1D) == self.num_of_points)

    def test_distribution_of_weights(self):
        #TODO: LIST
        # 1. Read Paper
        # 2. Generate my own samples
        # 3. Test it against the programs
        pass

    def test_distribution_of_grids(self):
        #TODO: Same as above
        pass


if __name__ == "__main__":
    unittest.main()