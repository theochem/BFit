import copy

class BST():
    def __init__(self, index, weight):
        assert type(index) is int, "index is not an Integer: %r" % index
        assert type(weight) is float, "weight is not a float: %r" % weight
        assert index >= 0, "index should be greater than equal to zero: %r" % index
        self.root = Node(index, weight, None, None)

    def min(self, *args):
        if len(args) == 0:
            return self.min(self.root).key
        elif len(args) == 1 and isinstance(args[0], Node):
            if args[0].left == None:
                return args[0]
            return min(args[0].left)
        else:
            raise TypeError("Arguments should be nothing or (Node)")

    def put_grid_point(self, *args):
        if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], float):
            self.root = self.put_grid_point(args[0], args[1], self.root)
        elif len(args) == 3 and args[2] is None or isinstance(args[2], Node):
            key = args[0]
            value = args[1]
            node = args[2]
            if(node == None):
                return Node(key, value, None, None)

            if key < node.key:
                node.left = self.put_grid_point( key, value, node.left)
            elif key > node.key:
                node.right = self.put_grid_point( key, value, node.right)
            elif key == node.key:
                node.value += value
            else:
                return None
            return (node)
        else:
            raise TypeError('Type of Arguments should formatted as (int, float) '
                            'or (int, float, Node), but the input is %r' % [type(x) for x in args]  )

    def get(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            self.get(args[0], self.root)
        elif len(args) == 1 and isinstance(args[1], Node):
            key = args[0]
            node = args[1]

            if (key < self.root.key):
                return(self.get(node.left, key))
            elif (key > self.root.key):
                return(self.get(node.right, key))
            elif key == self.root.key:
                return(node)
            else:
                return(None)
        else:
            raise TypeError('Arguments should be of type (int) or (int, Node)')

    def deleteMin(self, *args):

        if len(args) == 0:
            self.root = self.delete_grid_point(self.root, None)

        elif len(args) == 1 and isinstance(args[0], Node) and (isinstance(args[1], type(None)) or isinstance(args[1], Node)):
            if args[0].left == None:
                return args[0].right, args[0]
            args[0].left, min = self.deleteMin(args[0].left, args[1])
            return args[0], min
        else:
            raise TypeError("Arguments should be nothing or (Node)")

    def delete_grid_point(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            self.delete_grid_point(args[0], self.root)
        elif len(args) == 2 and isinstance(args[1], Node) and isinstance(args[0], int):
            key, node = args
            if (node == None):
                return None
            if key < node.key:
                node.left = self.delete_grid_point(node.left, key)
            elif key > node.key:
                node.right = self.delete_grid_point(node.right, key)
            else:
                if node.right == None:
                    return node.left
                elif node.left == None:
                    return node.right
                t = copy.deepcopy(node)
                node = self.min(t.right)
                node.right = self.deleteMin(t.right)[0]
                node.left = t.left

            return node


        else:
            raise TypeError('Type of Arguments should be either (int) or (int, Node)')

    def printNode(self, node):
        if node.left != None:
            self.printNode(node.left)
        if node.right != None:
            self.printNode(node.right)
        print(node.key)

    def __str__(self):
        return str(self.printNode(self.root))

class Node():
    def __init__(self, index, weight, left, right):
        assert type(index) is int, "Key is not an integer: %r" % index
        assert type(weight) is float, "Value is not a float: %r" % weight
        assert left is None or isinstance(left, Node), "Left is not an None/Node: %r" % left
        assert right is None or isinstance(right, Node), "Right is not an None/Node: %r" % right

        self.key = index
        self.value = weight
        self.left = left
        self.right = right