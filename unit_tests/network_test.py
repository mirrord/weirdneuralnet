import unittest
import weirdneuralnet.network as wnn
import cupy as np

class TestNodes(unittest.TestCase):
    # TODO: move to checking against raw matmult
    def test_predictV2(self):
        node_params = [
            {"x": 10, "y": 10, "activation": "sigmoid", "input": True},
            {
                "x": 10,
                "y": 10,
                "activation": "sigmoid",
            },
            {"x": 10, "y": 10, "activation": "sigmoid", "output": True},
        ]
        edges = [(0, 1), (1, 2), (0, 2)]
        model1 = wnn.WeirdNetwork(node_params, edges)
        model2 = wnn.WeirdNetworkV2(node_params, edges)
        model2.nodes[0].weight = model1.nodes[0].weight
        model2.nodes[1].weight = model1.nodes[1].weight
        model2.nodes[2].weight = model1.nodes[2].weight
        model2.nodes[0].bias = model1.nodes[0].bias
        model2.nodes[1].bias = model1.nodes[1].bias
        model2.nodes[2].bias = model1.nodes[2].bias

        input = np.random.randn(1, 10)
        predictions = np.array_equal(model1.predict(input), model2.predict(input))
        self.assertTrue(predictions)

        error_signal = np.random.randn(1, 10)
        bup1, wup1 = model1.backpropagate(error_signal)
        bup2, wup2 = model2.backpropagate(error_signal)
        for i in [0, 1]:
            bp_weights = np.array_equal(wup1[i], wup2[i])
            bp_biases = np.array_equal(bup1[i], bup2[i])
            self.assertTrue(bp_weights)
            self.assertTrue(bp_biases)
