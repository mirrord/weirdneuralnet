

import cupy as np

import unittest
#from weirdneuralnet.network import WeirdNetwork
from weirdneuralnet.node import *
from weirdneuralnet.node_utils import *
from weirdneuralnet.datasets import *

class TestNodes(unittest.TestCase):
    def test_nodev2_feed(self):
        n1 = Node(10,10,"sigmoid")
        n2 = NeuralNode(10,10,"sigmoid")
        n2.weight = n1.weight
        n2.bias = n1.bias

        input = np.random.randn(1, 10)
        self.assertTrue( np.array_equal( n1.feed(input), n2.feed({0:[input]}) ) )

        error_signal = np.random.randn(1, 10)
        n10, n11, n12 = n1.backfeed(error_signal)
        n20, n21, n22 = n2.backfeed(error_signal)
        self.assertTrue( np.array_equal(n10, n20) )
        self.assertTrue( np.array_equal(n11, n21) )
        self.assertTrue( np.array_equal(n12, n22) )

if __name__ == '__main__':
    unittest.main()