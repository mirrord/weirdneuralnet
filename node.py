import numpy as np
from node_utils import *

class Node():
    def __init__(self, input_dim, output_dim, activation):
        self.bias = np.random.randn(output_dim, 1)
        self.weight = np.random.randn(output_dim, input_dim)
        self.activate, self.backtivate = ACTIVATIONS.get(activation, no_activation)
        self.output = None
        self.input = None
        self.z = None

    def feed(self, input):
        #TODO:
        # add linear synapsing
        # add normalization
        self.input = input
        self.z = np.dot(input, self.weight) + self.bias
        self.output = self.activate(self.z)
        return self.output

    def backfeed(self, de_dz_foward):
        delta_bias = de_dz_foward * self.backtivate(self.z)
        delta_weight = np.dot(delta_bias, self.input.T)
        return delta_bias, delta_weight, np.dot(self.weight, delta_bias) #last is de_dz for next layer down

    def update(self, delta_bias, delta_weight):
        self.bias += delta_bias
        self.weight += delta_weight


    
