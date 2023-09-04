# import numpy as np
import cupy as np
from .node_utils import *
from typing import Dict


class NodeFeedException(Exception):
    pass


class BaseNode:
    def __init__(self, num_input_sites):
        self.num_input_sites = num_input_sites
        self.clear_history()

    def __str__(self):
        return "BaseNode"

    def set_dims(self):
        return

    def shape(self):
        return (0,)

    def clear_history(self):
        """Clear the cache, including input, output, and function pointers."""
        self.output = None
        self.inputs = {i: [] for i in range(self.num_input_sites)}
        self.z = None

    def get_output(self):
        """Retrieve the last-calculated output of this node, or 0 if never activated."""
        return self.output

    def does_update(self):
        return True

    # virtual
    def _fire(self):
        """calculates feed-foward output"""
        self.output = sum(self.inputs)
        return self.output

    def _synapse(self, inputs):
        """standard summation synapse function"""
        return [sum(site_inputs) for _, site_inputs in inputs.items()]

    def feed(self, inputs: Dict):
        self.inputs = self._synapse(inputs)
        return self._fire()

    # virtual
    def reload(self):
        return

    # virtual
    def backfeed(self, de_dz_foward: np.array):
        return 0, 0, [de_dz_foward]

    # virtual
    def update(self, delta_bias: np.array, delta_weight: np.array):
        """Update the weights and biases by subtraction."""
        return


class NeuralNode(BaseNode):
    def __init__(
        self, input_dim: int, output_dim: int, activation: str, normalize: str = ""
    ):
        """Construct a Node.
        Inputs:
            input_dim - the size of the input vector expected
            output_dim - the size of the output vector to be calculated
            activation - string name of activation function to use
            normalize - string name of normalization function to use
        """
        # TODO: impl more initialization strats?
        # currently using Xavier
        super().__init__(1)
        self.bias = np.random.randn(1, output_dim)
        self.weight = np.random.randn(input_dim, output_dim) * np.sqrt(
            1 / (input_dim + output_dim)
        )
        self.activate, self.backtivate = ACTIVATIONS.get(
            activation, (no_activation, no_activation)
        )
        self.normalize_label, self.normalize = normalize, NORMALIZATIONS.get(
            normalize, nonorm
        )
        self.activation_label = activation

    def shape(self):
        return self.weight.shape

    def reload(self):
        """Re-establish function pointers from stored function names."""
        self.activate, self.backtivate = ACTIVATIONS.get(
            self.activation_label, (no_activation, no_activation)
        )
        self.normalize = NORMALIZATIONS.get(self.normalize_label, nonorm)

    def clear_history(self):
        """Clear the cache, including input, output, and function pointers."""
        super().clear_history()
        self.activate = None
        self.backtivate = None
        self.normalize = None

    def _fire(self):
        """Calculate the foward activation with some input.
        Inputs:
            input - the input matrix, which must be of shape (num features, num samples)
        """
        if self.normalize_label:
            self.inputs[0] = self.normalize(self.inputs[0])
        self.z = np.dot(self.inputs[0], self.weight) + self.bias
        self.output = self.activate(self.z)
        return self.output

    def backfeed(self, de_dz_foward: np.array):
        """Calculate the gradient descent signal from the backpropogated error signal.
        Inputs:
            de_dz_forward - the next layers' error signal
        """
        delta_bias = de_dz_foward * self.backtivate(self.z)
        delta_weight = np.dot(self.inputs[0].T, delta_bias)
        return (
            delta_bias,
            delta_weight,
            [np.dot(delta_bias, self.weight.T)],
        )  # last is de_dz for next layer down

    def update(self, delta_bias: np.array, delta_weight: np.array):
        """Update the weights and biases by subtraction."""
        self.bias -= delta_bias
        self.weight -= delta_weight


class DualInputNode(BaseNode):
    def __init__(self, activation: str, normalize: str = ""):
        """Construct a Node.
        Inputs:
            activation - string name of activation function to use
            normalize - string name of normalization function to use
        """
        super().__init__(2)
        self.activate, self.backtivate = ACTIVATIONS.get(
            activation, (no_activation, no_activation)
        )
        self.normalize_label, self.normalize = normalize, NORMALIZATIONS.get(
            normalize, nonorm
        )
        self.activation_label = activation
        self._shape = None

    def set_dims(self, dims):
        self._shape = dims

    def shape(self):
        return self._shape

    def does_update(self):
        return False

    def reload(self):
        """Re-establish function pointers from stored function names."""
        self.activate, _ = ACTIVATIONS.get(
            self.activation_label, (no_activation, no_activation)
        )
        self.normalize = NORMALIZATIONS.get(self.normalize_label, nonorm)

    def clear_history(self):
        """Clear the cache, including input, output, and function pointers."""
        super().clear_history()
        self.activate = None
        self.normalize = None

    def _fire(self):
        """Calculate the foward activation with some input.
        Inputs:
            input - the input matrices, which must be of shape (num features, num samples)
        """
        if self.normalize_label:
            self.inputs[0] = self.normalize(self.inputs[0])
            self.inputs[1] = self.normalize(self.inputs[1])
        self.z = self.inputs[0] * self.inputs[1]
        self.output = self.activate(self.z)
        return self.output

    def backfeed(self, de_dz_foward: np.array):
        """Calculate the gradient descent signal from the backpropogated error signal.
        Inputs:
            de_dz_forward - the next layers' error signal
        """
        db = de_dz_foward * self.backtivate(self.z)
        de_dz = [
            db * self.inputs[1],
            db * self.inputs[0],
        ]
        return 0, 0, de_dz


class DelayNode(BaseNode):
    def __init__(self, dims=None):
        super().__init__(1)
        self._next_output = None
        if dims:
            self.output = np.zeros(dims)

    def get_output(self):
        return 0 if self.output == None else self.output

    def shape(self):
        return None if self.output == None else self.output.shape

    def set_dims(self, dims):
        if self.output == None:
            self.output = np.zeros(dims)

    def does_update(self):
        return False

    def clear_history(self):
        super().clear_history()
        self._next_output = None

    def _fire(self):
        if self.input[0].shape[1] > 1:
            # batch
            self.output = np.zeros(self.inputs[0].shape)
            self.output[1:] = self.inputs[0][:-1]
        else:
            # single
            self.output = self.next_output
            self.next_output = self.inputs[0]
        return self.output

    def backfeed(self, de_dz_foward: np.array):
        return 0, 0, [0]


class Convolution2DNode(BaseNode):
    def __init__(self, kernel_dims):
        super().__init__(1)
        self._filter = np.randn(*kernel_dims) / kernel_dims[0]*kernel_dims[1]

    def __str__(self):
        return f"ConvolutionNode with kernel ({self._filter.shape})"

    def _fire(self):
        """calculates feed-foward output"""
        filter_height, filter_width = self._filter.shape
        height, width = self.inputs[0]
        for i in range(height-(filter_height-1)):
            for j in range(width-(filter_width-1)):
                # how do you batch this???
        return self.output

    # taken from
    def _getWindows(self, input, output_size, kernel_size, padding=0, stride=1, dilate=0):
        working_input = input
        working_pad = padding
        # dilate the input if necessary
        if dilate != 0:
            working_input = np.insert(
                working_input, range(1, input.shape[2]), 0, axis=2)
            working_input = np.insert(
                working_input, range(1, input.shape[3]), 0, axis=3)

        # pad the input if necessary
        if working_pad != 0:
            working_input = np.pad(working_input, pad_width=(
                (0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

        in_b, in_c, out_h, out_w = output_size
        out_b, out_c, _, _ = input.shape
        batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

        return np.lib.stride_tricks.as_strided(
            working_input,
            (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
            (batch_str, channel_str, stride * kern_h_str,
             stride * kern_w_str, kern_h_str, kern_w_str)
        )

    def reload(self):
        return

    def backfeed(self, de_dz_foward: np.array):
        return 0, 0, [de_dz_foward]

    def update(self, delta_bias: np.array, delta_weight: np.array):
        """Update the weights and biases by subtraction."""
        return
