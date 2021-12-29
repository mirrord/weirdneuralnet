

import cupy as np

import unittest
#from weirdneuralnet.network import WeirdNetwork
from weirdneuralnet.node import *
from weirdneuralnet.node_utils import *
from weirdneuralnet.datasets import *

class TestNodes(unittest.TestCase):
    def nodev2_feed(self):
        n1 = Node(10,10,"sigmoid")
        n2 = NeuralNode(10,10,"sigmoid")
        n2.weight = n1.weight
        n2.bias = n2.bias

        input = np.random.randn(1, 10)
        self.assertTrue(np.equal(n1.feed(input), n2.feed({0:[input]})))

if __name__ == '__main__':
    unittest.main()