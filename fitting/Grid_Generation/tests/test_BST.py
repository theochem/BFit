from nose.tools import assert_equals, assert_not_equal
import unittest
from fitting.grid_generation.BST import BST, Node
import numpy as np
from nose.tools import with_setup

########### TODO LIST
### TODO - Ask Paul about the same list size for test_lt
### TODO - If above is true, add tests for list4

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


class Test_BST_Int_Key_Values(unittest.TestCase):
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

    def test_check_min(self):
        mins = self.binary_tree.min()
        assert_equals(mins.key, 0)

    def test_delete_min(self):
        for x in range(0, 9):
            self.binary_tree.deleteMin()
            assert_equals(self.binary_tree.min().key, x + 1)
        assert_not_equal(self.binary_tree.min().key, 11)

    def test_delete_grid_pt(self):
        #TODO: THIS
        pass



class Test_BST_Comparision_Operators(unittest.TestCase):
    def setUp(self):
        self.list1 = [1., 2., 3., 4.]
        self.list2 = [2., 3., 4., 5.]
        self.list3 = [1., 2., 3., 4.]
        self.list4 = [5., 6.]

        self.Node1 = Node(self.list1, 1.0, None, None)
        self.Node2 = Node(self.list2, 1.0, None, None)
        self.Node3 = Node(self.list3, 1.0, None, None)
        self.Node4 = Node(self.list4, 1.0, None, None)

    def test_eq(self):
        expected_results_1_2 = False
        result_1_2 = Node.comparing_two_lists(self.list1, self.list2, "=")
        assert expected_results_1_2 == result_1_2

        expected_results_1_3 = True
        result_1_3 = Node.comparing_two_lists(self.list1, self.list3, "=")
        assert expected_results_1_3 == result_1_3

        expected_results_1_4 = False
        assert expected_results_1_4 == Node.compare_two_elements(self.list1, self.list4, "=")

    def test_eq_through_nodes(self):
        assert True == (self.Node1 == self.Node3)
        assert False == (self.Node1 == self.Node2)
        assert False == (self.Node1 == self.Node4)


    def test_lt(self):
        assert True == Node.comparing_two_lists(self.list1, self.list2, "<")
        assert False == Node.comparing_two_lists(self.list1, self.list3, "<")
        #assert True == Node.comparing_two_lists(self.list1, self.list4, "<")

        assert False == (self.Node1 < self.Node3)
        assert True == (self.Node1 < self.Node2)
        assert False == (self.Node2 < self.Node1)

    def test_gt(self):
        assert False == Node.comparing_two_lists(self.list1, self.list2, ">")
        assert True == Node.comparing_two_lists(self.list2, self.list1, ">")
        assert False == Node.comparing_two_lists(self.list1, self.list3, ">")

        assert False == (self.Node1 > self.Node2)
        assert True == (self.Node2 > self.Node1)
        assert False == (self.Node1 > self.Node3)


    def test_le(self):
        assert True == Node.comparing_two_lists(self.list1, self.list2, "<=")
        assert False == Node.comparing_two_lists(self.list2, self.list1, "<=")
        assert True == Node.comparing_two_lists(self.list1, self.list3, "<=")

        assert True == (self.Node1 <= self.Node2)
        assert True == (self.Node1 <= self.Node3)
        assert False == (self.Node2 <= self.Node1)

    def test_ge(self):
        assert False == Node.comparing_two_lists(self.list1, self.list2, ">=")
        assert True == Node.comparing_two_lists(self.list2, self.list1, ">=")
        assert True == Node.comparing_two_lists(self.list1, self.list3, ">=")

        assert False == (self.Node1 >= self.Node2)
        assert True == (self.Node1 >= self.Node3)
        assert True == (self.Node2 >= self.Node1)


class Test_BST_Put_With_Lists(unittest.TestCase):
    def setUp(self):
        self.list1 = [2., 2., 3.]
        self.list2 = [1., 1., 2.]
        self.list3 = [3., 4., 5.]
        self.list4 = [0., 2., 3.]

        self.binary_tree = BST(self.list1, 1.0)

    def test_put(self):
        self.binary_tree.put_grid_point(self.list2, 2.0)
        assert self.binary_tree.root.left.key == self.list2
        assert self.binary_tree.root.left.value == 2.0
        assert self.binary_tree.root.right == None

        self.binary_tree.put_grid_point(self.list3, 4.0)
        assert self.binary_tree.root.right.key == self.list3
        assert not self.binary_tree.root.left.key == self.list3
        assert self.binary_tree.root.right.value == 4.0

        self.binary_tree.put_grid_point(self.list4, 5.0)
        assert self.binary_tree.root.left.left.key == self.list4
        assert self.binary_tree.root.left.left.value == 5.0

class Test_BST_Methods_With_Lists(unittest.TestCase):
    def setUp(self):
        self.index = [8., 6., 7.]
        self.weight = 223.0
        self.binary_tree = BST(self.index, self.weight)
        for x in range(0, 10):
            self.binary_tree.put_grid_point( [int(np.random.random() * 500) for x in range(0, 3)], np.random.random() )
            self.binary_tree.put_grid_point([x, x + 1, x + 2], float(x + 1))


    def test_min(self):
        assert self.binary_tree.min().key == [0., 1., 2.]
        assert self.binary_tree.min().value == 1.

    def test_get(self):
        get_list = [3., 4., 5.]
        assert self.binary_tree.get([3., 4. ,5.]).key == get_list
        assert self.binary_tree.get([3., 4., 5.]).value == 4.
        assert self.binary_tree.get([10., 11., 12.]) == None

    def test_delete_min(self):
        self.binary_tree.deleteMin()
        assert not self.binary_tree.min().key == [0., 1., 2.]
        assert not self.binary_tree.min().value == 1.
        assert self.binary_tree.min().key == [1., 2., 3.]
        assert self.binary_tree.min().value == 2.

    def test_delete_grid_point(self):
        delete_point = [7., 8., 9.]
        self.binary_tree.delete_grid_point(delete_point)

        assert self.binary_tree.get([7., 8., 9.]) == None
        delete_root = [8., 6., 7.]
        self.binary_tree.delete_grid_point(delete_root)
        assert self.binary_tree.root.key == [8., 9., 10.]
        assert self.binary_tree.root.left.key == [0., 1., 2.]


if __name__ == "__main__":
    unittest.main()