
#import numpy as np
import cupy as np
from node_utils import *

class Node():
    def __init__(self, input_dim, output_dim, activation):
        #TODO: impl initialization strats
        # currently using Xavier
        self.bias = np.random.randn(output_dim, 1)
        self.weight = np.random.randn(output_dim, input_dim) * np.sqrt(1/(input_dim+output_dim))
        self.activate, self.backtivate = ACTIVATIONS.get(activation, no_activation)
        self.activation_label = activation
        self.output = None
        self.input = None
        self.z = None

    def clear_history(self):
        self.output = None
        self.input = None
        self.z = None
        self.activate = None
        self.backtivate = None

    def get_output(self):
        return self.output if self.output else 0

    def reload(self):
        self.activate, self.backtivate = ACTIVATIONS.get(self.activation_label, no_activation)

    def shape(self):
        return self.weight.shape

    def feed(self, input):
        #TODO:
        # add linear synapsing
        # add normalization
        self.input = input
        self.z = np.dot(self.weight, input) + self.bias
        #print(f"activation: {self.activate}")
        self.output = self.activate(self.z)
        #print(f"δ({self.weight.shape} . {input.shape} + {self.bias.shape}) = {self.output.shape}")
        # print(f"weight: {self.weight}\n")
        # print(f"input: {input}\n")
        # print(f"output: {self.output}\n")
        return self.output

    def backfeed(self, de_dz_foward):
        #print(f"de/dz: {de_dz_foward.shape}")
        delta_bias = de_dz_foward * self.backtivate(self.z)
        #print(f"delta bias: {delta_bias.shape} vs my bias: {self.bias.shape}")
        #print(f"calculating: bias {delta_bias.shape} . input.T {self.input.T.shape}")
        delta_weight = np.dot(delta_bias, self.input.T)
        #print(f"delta weight: {delta_weight.shape} vs my weight: {self.weight.shape}")
        #print(f"db: {delta_bias.shape}, dw: {delta_weight.shape}")
        return delta_bias, delta_weight, np.dot(self.weight.T, delta_bias) #last is de_dz for next layer down

    def update(self, delta_bias, delta_weight):
        self.bias -= delta_bias
        self.weight -= delta_weight


    
