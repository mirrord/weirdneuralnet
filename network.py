

from node import Node
from node_utils import *
#import numpy as np
import cupy as np

from pickle import Pickler, Unpickler


class WeirdNetwork():
    def __init__(self,
                node_params,
                edges,
                error_cost_func="diff_squares",
                learning_rate=0.01,
                regularize=(None, 0)):
        # node_params should be a list of dicts of node parameters
        # edges should be a list of node index tuples denoting connections between nodes
        self.nodes = []
        self.input_indices = []
        self.feed_indices = {}
        self.backfeed_indices = {}
        self.output_node = 0
        self.learning_rate = learning_rate
        self.input, self.output = None, None
        self.regularize_params, self.regularize = regularize, REGULARIZATIONS.get(regularize[0], noreg)(regularize[1])
        self.cost, self.cost_deriv = COSTS.get(error_cost_func, (diff_squares, ddiff_squares))
        for idx, param in enumerate(node_params):
            self.feed_indices[idx] = [edge[1] for edge in edges if edge[0]==idx]
            self.backfeed_indices[idx] = [edge[0] for edge in edges if edge[1]==idx]
            #print(f"backfeed idxs {idx}: {self.backfeed_indices[idx]}")
            self.nodes.append(Node(param['x'], param['y'], param['activation']))
            if param.get('output', False):
                self.output_node = idx
            if param.get('input', False):
                self.input_indices.append(idx)
        self.edges = edges


    def __str__(self):
        return f"<WeirdNetwork size={len(self.nodes)} nodes>"

    #TODO: this is insecure and doesn't really work.
    #   implement a weirdnet-specific load & save.
    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            u = Unpickler(f)
            model = u.load()
        for node in model.nodes:
            node.reload()
        model.regularize = REGULARIZATIONS.get(model.regularize_params[0], noreg)(model.regularize_params[1])
        return model

    def save(self, fname, keep_history=False):
        if not keep_history:
            self.regularize = None
            for node in self.nodes:
                node.clear_history()
        with open(fname, 'wb') as f:
            p = Pickler(f)
            return p.dump(self)

    def compile(self):
        # only works for linear networks
        to_traverse = []
        to_traverse.extend(self.input_indices)
        for idx in self.input_indices:
            to_traverse.extend(self.feed_indices[idx])
        for idx in to_traverse:
            to_traverse.extend([fidx for fidx in self.feed_indices[idx] if fidx not in to_traverse])
        if self.output_node in to_traverse:
            weights = [self.nodes[nidx].weight for nidx in to_traverse]
            biases = [self.nodes[nidx].bias for nidx in to_traverse]
            return weights, biases
        raise Exception("Output node is not fed")

    def predict(self, input, debinarize=False):
        '''input shape must be (features, samples)'''
        outputs = {}
        to_traverse = []
        #print(f"feed idxs: {self.feed_indices}")
        #print(f"backfeed idxs: {self.backfeed_indices}")
        # print(f"predicting... {input.shape}")
        for idx in self.input_indices:
            # print(f"feeding input node {idx}...")
            outputs[idx] = self.nodes[idx].feed(input)
            to_traverse.extend(self.feed_indices[idx])
        # print(f"initial propogation targets: {to_traverse}")
        for idx in to_traverse:
            if idx not in outputs:
                #find outputs this node wants as input
                inputs = [self.nodes[i].get_output() for i in self.backfeed_indices[idx]]
                #print(f"feeding node {idx} with outputs from: {[i for i in self.backfeed_indices[idx] if i in outputs]}")
                #"synapse" them together
                this_input = sum(inputs)
                outputs[idx] = self.nodes[idx].feed(this_input)
                to_traverse.extend([fidx for fidx in self.feed_indices[idx] if fidx not in outputs])
        if self.output_node in outputs:
            #print(f"taking output from output node {self.output_node}")
            #print(f"prediction output: {outputs[self.output_node].shape}")
            if debinarize:
                return outputs[self.output_node].argmax(axis=0)
            return outputs[self.output_node]
        raise Exception("Output node is not fed")

    def get_last_output(self):
        return self.nodes[self.output_node].output

    ##TODO: backprop, train, minibatch train, etc
    def backpropagate(self, input, exp_output):
        num_sample = input.shape[1]
        predicted_output = self.predict(input)
        # print(f"output: {predicted_output.shape}")
        backfeed = {}
        error_delta = self.cost_deriv(predicted_output, exp_output)
        # print(f"dE {error_delta.shape} = dC( {predicted_output.shape}, {exp_output.shape} )")
        # print(f"(backprop) before backfeed: {self.backfeed_indices}")
        backfeed[self.output_node] = self.nodes[self.output_node].backfeed(error_delta)
        # print(f"(backprop) after backfeed: {self.backfeed_indices}")
        to_traverse = self.backfeed_indices.get(self.output_node, []).copy()
        # print(f"(backprop) after to_traverse: {self.backfeed_indices}")
        for idx in to_traverse:
            error_signal_components = [backfeed.get(oidx, (0,0,0)) for oidx in self.feed_indices[idx]]
            de = sum([i[2] for i in error_signal_components]) #I think this actually requires a derivative of the synapse func
            db, dw, de = self.nodes[idx].backfeed(de)
            if idx not in backfeed:
                backfeed[idx] = db, dw, de
            else:
                backfeed[idx][0]+=db
                backfeed[idx][1]+=dw
                backfeed[idx][2]+=de

            for jdx in self.backfeed_indices[idx]:
                if jdx not in to_traverse:
                    to_traverse.append(jdx)

        bup = {i:np.sum(b[0],1,keepdims=True)/num_sample for i,b in backfeed.items()}
        wup = {i:w[1]/num_sample for i,w in backfeed.items()}
        return bup, wup

    def evaluate(self, input, exp_out):
        return self.cost(self.predict(input),exp_out)

    def train(self, input, exp_output, epochs):
        #TODO: adapt to use a batch size
        cost_history = []
        for i in range(epochs):
            print(f"epoch {i}...") #TODO: use progress bar instead
            shuffle_in_unison(input.T, exp_output.T)
            bup, wup = self.backpropagate(input, exp_output)
            cost_history.append(self.cost(self.get_last_output(), exp_output))
            #update
            for idx, node in enumerate(self.nodes):
                if idx in wup:
                    update_weights = self.learning_rate*wup[idx]
                    if self.regularize_params[0] is not None:
                        update_weights+=self.regularize(wup)
                    node.update(self.learning_rate*bup[idx], update_weights)

        return cost_history










### This code taken from github. no license given.
#### Libraries
# Standard library
# import random

# # Third-party libraries


# class Network(object):

#     def __init__(self, sizes):
#         """The list ``sizes`` contains the number of neurons in the
#         respective layers of the network.  For example, if the list
#         was [2, 3, 1] then it would be a three-layer network, with the
#         first layer containing 2 neurons, the second layer 3 neurons,
#         and the third layer 1 neuron.  The biases and weights for the
#         network are initialized randomly, using a Gaussian
#         distribution with mean 0, and variance 1.  Note that the first
#         layer is assumed to be an input layer, and by convention we
#         won't set any biases for those neurons, since biases are only
#         ever used in computing the outputs from later layers."""
#         self.num_layers = len(sizes)
#         self.sizes = sizes
#         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x)
#                         for x, y in zip(sizes[:-1], sizes[1:])]

#     def feedforward(self, a):
#         """Return the output of the network if ``a`` is input."""
#         for b, w in zip(self.biases, self.weights):
#             a = sigmoid(np.dot(w, a)+b)
#         return a

#     def SGD(self, training_data, epochs, mini_batch_size, eta,
#             test_data=None):
#         """Train the neural network using mini-batch stochastic
#         gradient descent.  The ``training_data`` is a list of tuples
#         ``(x, y)`` representing the training inputs and the desired
#         outputs.  The other non-optional parameters are
#         self-explanatory.  If ``test_data`` is provided then the
#         network will be evaluated against the test data after each
#         epoch, and partial progress printed out.  This is useful for
#         tracking progress, but slows things down substantially."""

#         training_data = list(training_data)
#         n = len(training_data)

#         if test_data:
#             test_data = list(test_data)
#             n_test = len(test_data)

#         for j in range(epochs):
#             random.shuffle(training_data)
#             mini_batches = [
#                 training_data[k:k+mini_batch_size]
#                 for k in range(0, n, mini_batch_size)]
#             for mini_batch in mini_batches:
#                 self.update_mini_batch(mini_batch, eta)
#             if test_data:
#                 print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
#             else:
#                 print("Epoch {} complete".format(j))

#     def update_mini_batch(self, mini_batch, eta):
#         """Update the network's weights and biases by applying
#         gradient descent using backpropagation to a single mini batch.
#         The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
#         is the learning rate."""
#         nabla_b = [np.zeros(b.shape) for b in self.biases]
#         nabla_w = [np.zeros(w.shape) for w in self.weights]
#         for x, y in mini_batch:
#             delta_nabla_b, delta_nabla_w = self.backprop(x, y)
#             nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
#             nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
#         self.weights = [w-(eta/len(mini_batch))*nw
#                         for w, nw in zip(self.weights, nabla_w)]
#         self.biases = [b-(eta/len(mini_batch))*nb
#                        for b, nb in zip(self.biases, nabla_b)]

#     def backprop(self, x, y):
#         """Return a tuple ``(nabla_b, nabla_w)`` representing the
#         gradient for the cost function C_x.  ``nabla_b`` and
#         ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
#         to ``self.biases`` and ``self.weights``."""
#         nabla_b = [np.zeros(b.shape) for b in self.biases]
#         nabla_w = [np.zeros(w.shape) for w in self.weights]
#         # feedforward
#         activation = x
#         activations = [x] # list to store all the activations, layer by layer
#         zs = [] # list to store all the z vectors, layer by layer
#         for b, w in zip(self.biases, self.weights):
#             z = np.dot(w, activation)+b
#             zs.append(z)
#             activation = sigmoid(z)
#             activations.append(activation)
#         # backward pass
#         delta = self.cost_derivative(activations[-1], y) * \
#             sigmoid_prime(zs[-1])
#         nabla_b[-1] = delta
#         nabla_w[-1] = np.dot(delta, activations[-2].transpose())
#         # Note that the variable l in the loop below is used a little
#         # differently to the notation in Chapter 2 of the book.  Here,
#         # l = 1 means the last layer of neurons, l = 2 is the
#         # second-last layer, and so on.  It's a renumbering of the
#         # scheme in the book, used here to take advantage of the fact
#         # that Python can use negative indices in lists.
#         for l in range(2, self.num_layers):
#             z = zs[-l]
#             sp = sigmoid_prime(z)
#             delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
#             nabla_b[-l] = delta
#             nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
#         return (nabla_b, nabla_w)

#     def evaluate(self, test_data):
#         """Return the number of test inputs for which the neural
#         network outputs the correct result. Note that the neural
#         network's output is assumed to be the index of whichever
#         neuron in the final layer has the highest activation."""
#         test_results = [(np.argmax(self.feedforward(x)), y)
#                         for (x, y) in test_data]
#         return sum(int(x == y) for (x, y) in test_results)

#     def cost_derivative(self, output_activations, y):
#         """Return the vector of partial derivatives \partial C_x /
#         \partial a for the output activations."""
#         return (output_activations-y)

# #### Miscellaneous functions
# # def sigmoid(z):
# #     """The sigmoid function."""
# #     return 1.0/(1.0+np.exp(-z))

# def sigmoid_prime(z):
#     """Derivative of the sigmoid function."""
#     return sigmoid(z)*(1-sigmoid(z))