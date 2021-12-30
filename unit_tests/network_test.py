
import unittest
from weirdneuralnet.network import *


class TestNodes(unittest.TestCase):
    def test_predictV2(self):
        node_params =[
                {
                    'x':10,
                    'y':10,
                    'activation': 'sigmoid',
                    'input':True
                },
                {
                    'x':10,
                    'y':10,
                    'activation': 'sigmoid',
                    'output':True
                }
            ]
        edges = [
            (0,1),
        ]
        model1 = WeirdNetwork(node_params, edges)
        model2 = WeirdNetworkV2(node_params, edges)
        model2.nodes[0].weight = model1.nodes[0].weight
        model2.nodes[1].weight = model1.nodes[1].weight
        model2.nodes[0].bias = model1.nodes[0].bias
        model2.nodes[1].bias = model1.nodes[1].bias

        input = np.random.randn(1, 10)
        self.assertTrue( np.array_equal( 
            model1.predict(input),
            model2.predict(input)
         ) )

        error_signal = np.random.randn(1, 10)
        bup1, wup1 = model1.backpropagate(error_signal)
        bup2, wup2 = model2.backpropagate(error_signal)
        for i in [0,1]:
            self.assertTrue( np.array_equal(
                wup1[i], wup2[i]
            ) )
            self.assertTrue( np.array_equal(
                bup1[i], bup2[i]
            ) )
