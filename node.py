import numpy as np
from node_utils import *

class Node():
    def __init__(self, input_dim, output_dim, activation):
        self.bias = np.random.randn(1, output_dim)
        self.weight = np.random.randn(input_dim, output_dim)
        self.activate, self.backtivate = ACTIVATIONS.get(activation, no_activation)
        self.output = None
        self.input = None
        self.z = None

    def feed(self, input):
        #TODO:
        # add linear synapsing
        # add normalization
        self.input = input
        self.z = np.dot(self.weight, input) + self.bias
        self.output = self.activate(self.z)
        return self.output

    def backfeed(self, de_dz_foward):
        print(f"de/dz: {de_dz_foward.shape}")
        delta_bias = de_dz_foward * self.backtivate(self.z)
        print(f"delta bias: {delta_bias.shape} vs my bias: {self.bias.shape}")
        delta_weight = np.dot(delta_bias, self.input.T)
        print(f"delta weight: {delta_weight.shape} vs my weight: {self.weight.shape}")
        return delta_bias, delta_weight, np.dot(self.weight.T, delta_bias) #last is de_dz for next layer down

    def update(self, delta_bias, delta_weight):
        self.bias += delta_bias
        self.weight += delta_weight


    
