from nose.tools import assert_equals, assert_not_equal
import unittest
from fitting.grid_generation.BST import BST, Node
import numpy as np
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
    def setUp(self):
        self.index = 5
        self.weight = 223.0
        self.binary_tree = BST(self.index, self.weight)
        for x in range(0, 10):
            self.binary_tree.put_grid_point( int(np.random.random() * 500), np.random.random() )
            self.binary_tree.put_grid_point(x, float(x + 1))


    def test_get_grid_point(self):
        node_5 = self.binary_tree.get(5)

        assert_equals(node_5.key, 5)
        assert_equals(node_5.value,  229.0)

        node_9 = self.binary_tree.get(9)
        assert_equals(node_9.key, 9)
        assert_equals(node_9.value, 10)

        node_2 = self.binary_tree.get(2)
        assert_equals(node_2.key, 2)
        assert_equals(node_2.value, 3)

        node_0 = self.binary_tree.get(0)
        assert_equals(node_0.left, None)

class Test_BST_Min_Method(unittest.TestCase):
    def setUp(self):
        self.index = 5
        self.weight = 223.0
        self.binary_tree = BST(self.index, self.weight)
        for x in range(0, 10):
            self.binary_tree.put_grid_point( int(np.random.random() * 500), np.random.random() )
            self.binary_tree.put_grid_point(x, float(x + 1))

    def test_check_min(self):
        mins = self.binary_tree.min()
        assert_equals(mins.key, 0)


class Test_BST_Delete_Min(unittest.TestCase):
    def setUp(self):
       self.index = 5
       self.weight = 223.0
       self.binary_tree = BST(self.index, self.weight)
       for x in range(0, 10):
           self.binary_tree.put_grid_point( int(np.random.random() * 500), np.random.random() )
           self.binary_tree.put_grid_point(x, float(x + 1))

    def test_delete_min(self):
        for x in range(0, 9):
            self.binary_tree.deleteMin()
            assert_equals(self.binary_tree.min().key, x + 1)
        assert_not_equal(self.binary_tree.min().key, 11)


class Test_BST_Delete_GridPt(unittest.TestCase):
    def setUp(self):
       self.index = 5
       self.weight = 223.0
       self.binary_tree = BST(self.index, self.weight)
       for x in range(0, 10):
           self.binary_tree.put_grid_point( int(np.random.random() * 500), np.random.random() )
           self.binary_tree.put_grid_point(x, float(x + 1))

    def test_delete_grid_pt(self):
        #TODO: THIS
        pass


if __name__ == "__main__":
    unittest.main()