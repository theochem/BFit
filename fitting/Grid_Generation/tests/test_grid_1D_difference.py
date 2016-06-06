from nose.tools import assert_equals, assert_not_equal
import unittest
from fitting.grid_generation.BST import BST, Node
from nose.tools import with_setup


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
