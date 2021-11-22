import numpy as np

## activation functions
from scipy.special import expit as sigmoid
def dsigmoid(x):
    sig = sigmoid(x)
    return sig*(1-sig)

def no_activation(x):
    raise Exception("activation function not found or not implemented")

ACTIVATIONS ={
    "sigmoid": (sigmoid, dsigmoid),
}

## cost functions
def diff_squares(y, y_true):
    return np.square(y-y_true)
def ddiff_squares(y, y_true):
    return 2*(y-y_true)

def no_cost(x):
    raise Exception("cost function not found or not implemented")
COSTS = {
    "diff_squares": (diff_squares, ddiff_squares),
}
## regularization functions

## synapse functions

## normalization functions



