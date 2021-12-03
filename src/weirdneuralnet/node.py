
#import numpy as np
import cupy as np
from .node_utils import *

class Node():
    '''Represents a layer in conventional NN terms.

    Methods
    -------------
    clear_history
        clear the cache
    get_output
        retrieve the most recent activation output
    reload
        reinitialize important cache variables
    shape
        get the shape of the node's matrix
    feed
        calculate activation output
    backfeed
        calculate backprop signals
    update
        update the weights and biases'''

    def __init__(self,
                input_dim:int,
                output_dim:int,
                activation:str,
                normalize:str=''):
        '''Construct a Node.
        Inputs:
            input_dim - the size of the input vector expected
            output_dim - the size of the output vector to be calculated
            activation - string name of activation function to use
            normalize - string name of normalization function to use
        '''
        #TODO: impl initialization strats
        # currently using Xavier
        self.bias = np.random.randn(output_dim, 1)
        self.weight = np.random.randn(output_dim, input_dim) * np.sqrt(1/(input_dim+output_dim))
        self.activate, self.backtivate = ACTIVATIONS.get(activation, no_activation)
        self.normalize_label, self.normalize = normalize, NORMALIZATIONS.get(normalize, nonorm)
        self.activation_label = activation
        self.output = None
        self.input = None
        self.z = None

    def clear_history(self):
        '''Clear the cache, including input, output, and function pointers.'''
        self.output = None
        self.input = None
        self.z = None
        self.activate = None
        self.backtivate = None
        self.normalize = None

    def get_output(self):
        '''Retrieve the last-calculated output of this node, or 0 if never activated.'''
        return 0 if self.output is None else self.output

    def reload(self):
        '''Re-establish function pointers from stored function names.'''
        self.activate, self.backtivate = ACTIVATIONS.get(self.activation_label, no_activation)
        self.normalize = NORMALIZATIONS.get(self.normalize_label, nonorm)

    def shape(self):
        '''Get the shape of the weight matrix.'''
        return self.weight.shape

    def feed(self, input):
        '''Calculate the foward activation with some input.
        Inputs:
            input - the input matrix, which must be of shape (num features, num samples)
        '''
        if self.normalize_label:
            input = self.normalize(input)
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
        '''Calculate the gradient descent signal from the backpropogated error signal.
        Inputs:
            de_dz_forward - the next layers' error signal
        '''
        #print(f"de/dz: {de_dz_foward.shape}")
        delta_bias = de_dz_foward * self.backtivate(self.z)
        #print(f"delta bias: {delta_bias.shape} vs my bias: {self.bias.shape}")
        #print(f"calculating: bias {delta_bias.shape} . input.T {self.input.T.shape}")
        delta_weight = np.dot(delta_bias, self.input.T)
        #print(f"delta weight: {delta_weight.shape} vs my weight: {self.weight.shape}")
        #print(f"db: {delta_bias.shape}, dw: {delta_weight.shape}")
        return delta_bias, delta_weight, np.dot(self.weight.T, delta_bias) #last is de_dz for next layer down

    def update(self, delta_bias, delta_weight):
        '''Update the weights and biases by subtraction.'''
        self.bias -= delta_bias
        self.weight -= delta_weight


    
