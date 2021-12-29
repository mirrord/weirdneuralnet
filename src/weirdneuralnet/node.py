
#import numpy as np
import cupy as np
from .node_utils import *
from typing import Dict

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
        self.bias = np.random.randn(1, output_dim)
        self.weight = np.random.randn(input_dim, output_dim) * np.sqrt(1/(input_dim+output_dim))
        self.activate, self.backtivate = ACTIVATIONS.get(activation, (no_activation, no_activation))
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
        self.activate, self.backtivate = ACTIVATIONS.get(self.activation_label, (no_activation, no_activation))
        self.normalize = NORMALIZATIONS.get(self.normalize_label, nonorm)

    def shape(self):
        '''Get the shape of the weight matrix.'''
        return self.weight.shape

    def feed(self, input:np.array):
        '''Calculate the foward activation with some input.
        Inputs:
            input - the input matrix, which must be of shape (num features, num samples)
        '''
        if self.normalize_label:
            input = self.normalize(input)
        self.input = input
        self.z = np.dot(input, self.weight) + self.bias
        self.output = self.activate(self.z)
        #print(f"δ({self.weight.shape} . {input.shape} + {self.bias.shape}) = {self.output.shape}")
        # print(f"weight: {self.weight}\n")
        # print(f"input: {input}\n")
        # print(f"output: {self.output}\n")
        return self.output

    def backfeed(self, de_dz_foward:np.array):
        '''Calculate the gradient descent signal from the backpropogated error signal.
        Inputs:
            de_dz_forward - the next layers' error signal
        '''
        #print(f"de/dz: {de_dz_foward.shape}")
        delta_bias = de_dz_foward * self.backtivate(self.z)
        #print(f"delta bias: {delta_bias.shape} vs my bias: {self.bias.shape}")
        #print(f"calculating: bias {delta_bias.shape} . input.T {self.input.T.shape}")
        delta_weight = np.dot(self.input.T, delta_bias)
        #print(f"delta weight: {delta_weight.shape} vs my weight: {self.weight.shape}")
        #print(f"db: {delta_bias.shape}, dw: {delta_weight.shape}")
        return delta_bias, delta_weight, np.dot(delta_bias, self.weight.T) #last is de_dz for next layer down

    def update(self, delta_bias:np.array, delta_weight:np.array):
        '''Update the weights and biases by subtraction.'''
        self.bias -= delta_bias
        self.weight -= delta_weight


####################################
### Node System V2: under construction
####################################
class NodeFeedException(Exception):
    pass

class BaseNode():
    def __init__(self, num_input_sites):
        self.num_input_sites = num_input_sites
        self.clear_history()

    def __str__(self):
        return "BaseNode"

    def shape(self):
        return (0,)

    def clear_history(self):
        '''Clear the cache, including input, output, and function pointers.'''
        self.output = None
        self.inputs = {i:[] for i in range(self.num_input_sites)}
        self.z = None

    def get_output(self):
        '''Retrieve the last-calculated output of this node, or 0 if never activated.'''
        return 0 if self.output is None else self.output

    #virtual
    def _fire(self):
        '''calculates feed-foward output'''
        self.output = sum(self.inputs)
        return self.output

    def _synapse(self, inputs):
        '''standard summation synapse function'''
        return [sum(site_inputs) for _, site_inputs in inputs.items()]

    def feed(self, inputs:Dict):
        self.inputs = self._synapse(inputs)
        return self._fire()

    #virtual
    def reload(self):
        return

    #virtual
    def backfeed(self, de_dz_foward:np.array):
        return 0, 0, de_dz_foward

    #virtual
    def update(self, delta_bias:np.array, delta_weight:np.array):
        '''Update the weights and biases by subtraction.'''
        return


class NeuralNode(BaseNode):
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
        #TODO: impl more initialization strats?
        # currently using Xavier
        super().__init__(1)
        self.bias = np.random.randn(1, output_dim)
        self.weight = np.random.randn(input_dim, output_dim) * np.sqrt(1/(input_dim+output_dim))
        self.activate, self.backtivate = ACTIVATIONS.get(activation, (no_activation, no_activation))
        self.normalize_label, self.normalize = normalize, NORMALIZATIONS.get(normalize, nonorm)
        self.activation_label = activation

    def reload(self):
        '''Re-establish function pointers from stored function names.'''
        self.activate, self.backtivate = ACTIVATIONS.get(self.activation_label, (no_activation, no_activation))
        self.normalize = NORMALIZATIONS.get(self.normalize_label, nonorm)

    def clear_history(self):
        '''Clear the cache, including input, output, and function pointers.'''
        super().clear_history()
        self.activate = None
        self.backtivate = None
        self.normalize = None

    def _fire(self):
        '''Calculate the foward activation with some input.
        Inputs:
            input - the input matrix, which must be of shape (num features, num samples)
        '''
        if self.normalize_label:
            self.inputs[0] = self.normalize(self.inputs[0])
        self.z = np.dot(self.inputs[0], self.weight) + self.bias
        self.output = self.activate(self.z)
        return self.output

    def backfeed(self, de_dz_foward:np.array):
        '''Calculate the gradient descent signal from the backpropogated error signal.
        Inputs:
            de_dz_forward - the next layers' error signal
        '''
        delta_bias = de_dz_foward * self.backtivate(self.z)
        delta_weight = np.dot(self.inputs[0].T, delta_bias)
        return delta_bias, delta_weight, np.dot(delta_bias, self.weight.T) #last is de_dz for next layer down

    def update(self, delta_bias:np.array, delta_weight:np.array):
        '''Update the weights and biases by subtraction.'''
        self.bias -= delta_bias
        self.weight -= delta_weight

class DualNeuralNode(BaseNode):
    def __init__(self,
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
        super().__init__(2)
        self.activate, _ = ACTIVATIONS.get(activation, (no_activation, no_activation))
        self.normalize_label, self.normalize = normalize, NORMALIZATIONS.get(normalize, nonorm)
        self.activation_label = activation

    def reload(self):
        '''Re-establish function pointers from stored function names.'''
        self.activate, _ = ACTIVATIONS.get(self.activation_label, (no_activation, no_activation))
        self.normalize = NORMALIZATIONS.get(self.normalize_label, nonorm)

    def clear_history(self):
        '''Clear the cache, including input, output, and function pointers.'''
        super().clear_history()
        self.activate = None
        self.normalize = None

    def _fire(self):
        '''Calculate the foward activation with some input.
        Inputs:
            input - the input matrix, which must be of shape (num features, num samples)
        '''
        if self.normalize_label:
            self.inputs[0] = self.normalize(self.inputs[0])
            self.inputs[1] = self.normalize(self.inputs[1])
        self.z = np.dot(self.inputs[0], self.inputs[1].T)
        self.output = self.activate(self.z)
        return self.output

    def backfeed(self, de_dz_foward:np.array):
        '''Calculate the gradient descent signal from the backpropogated error signal.
        Inputs:
            de_dz_forward - the next layers' error signal
        '''
        #TODO: I'm not certain how this should be calculated, if at all
        return 0, 0, de_dz_foward #last is de_dz for next layer down