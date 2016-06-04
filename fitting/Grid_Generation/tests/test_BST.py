from nose.tools import assert_equals, assert_not_equal
import unittest
from fitting.grid_generation.BST import BST, Node
from nose.tools import with_setup

class Test_BST_Put_Method(unittest.TestCase):
    def setUp(self):
        self.index = 5
        self.weight = 5.0
        self.binary_tree = BST(self.index, self.weight)


    def teardown(self):
        print ("TestUM:teardown() after each test method")

    @classmethod
    def setup_class(cls):
        print ("setup_class() before any methods in this class")

    @classmethod
    def teardown_class(cls):
        print ("teardown_class() after any methods in this class")

    def test_put_grid_point_root(self):
        # Tests if the binary tree is set up right
        assert_equals(self.binary_tree.root.key, self.index)
        assert_equals(self.binary_tree.root.value, self.weight)
        assert_equals(self.binary_tree.root.left, None)
        assert_equals(self.binary_tree.root.right, None)



    def test_put_grid_point_smaller_input(self):
        self.binary_tree.put_grid_point(3, 5.0)

        assert_equals(self.binary_tree.root.left.key, 3)
        assert_equals(self.binary_tree.root.left.value, 5.0)
        assert_equals(self.binary_tree.root.left.right, None)
        assert_equals(self.binary_tree.root.left.left, None)

        assert_equals(self.binary_tree.root.right, None)
        assert_equals(self.binary_tree.root.key, self.index)
        assert_equals(self.binary_tree.root.value, self.weight)

    def test_put_grid_point_larger_input(self):
        self.binary_tree.put_grid_point(6, 10.0)
        self.binary_tree.put_grid_point(3, 12321.0)

        assert_equals(self.binary_tree.root.right.key, 6.0)
        assert_equals(self.binary_tree.root.right.value, 10.0)
        assert_equals(self.binary_tree.root.right.right, None)
        assert_equals(self.binary_tree.root.right.left, None)

        assert_equals(self.binary_tree.root.left.key, 3)
        assert_equals(self.binary_tree.root.left.value, 12321.0)
        assert_equals(self.binary_tree.root.left.right, None)
        assert_equals(self.binary_tree.root.left.left, None)

    def test_put_grid_point_smaller_input_in_a_tree(self):
        self.binary_tree.put_grid_point(6, 10.0)
        self.binary_tree.put_grid_point(3, 12321.0)
        self.binary_tree.put_grid_point(1, 12.0)
        self.binary_tree.put_grid_point(0, 10.0)
        self.binary_tree.put_grid_point(2, 123.0)

        assert_equals(self.binary_tree.root.left.left.left.key, 0)
        assert_equals(self.binary_tree.root.left.left.key, 1)
        assert_equals(self.binary_tree.root.left.key, 3)
        assert_equals(self.binary_tree.root.right.key, 6)
        assert_equals(self.binary_tree.root.left.left.right.key, 2)

    def test_put_grid_point_larger_input_in_a_tree(self):
        self.binary_tree.put_grid_point(7, 10.0)
        self.binary_tree.put_grid_point(3, 12321.0)
        self.binary_tree.put_grid_point(9, 10.0)
        self.binary_tree.put_grid_point(6, 10.0)

        assert_equals(self.binary_tree.root.right.key, 7)
        assert_equals(self.binary_tree.root.right.left.key, 6)
        assert_equals(self.binary_tree.root.right.right.key, 9)


class Test_BST_Get_Method(unittest.TestCase):
    def setup(self):
        print ("TestUM:setup() before each test method")

    def teardown(self):
        print ("TestUM:teardown() after each test method")

    @classmethod
    def setup_class(cls):
        print ("setup_class() before any methods in this class")


    @classmethod
    def teardown_class(cls):
        print ("teardown_class() after any methods in this class")

    def test_get_grid_point(self):
        pass




if __name__ == "__main__":
    unittest.main()