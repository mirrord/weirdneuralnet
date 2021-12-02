
#import numpy as np
import cupy as np

## activation functions
#from scipy.special import expit as sigmoid
sigmoid = np.ElementwiseKernel(
        'float64 x',
        'float64 y',
        'y = 1 / (1 + exp(-x))',
        'expit')
def dsigmoid(x):
    sig = sigmoid(x)
    return sig*(1-sig)

relu = np.ElementwiseKernel(
        'float64 x',
        'float64 y',
        'y = x * (x > 0)',
        'relu')
def drelu(x):
    return np.greater(x, 0).astype('float64')

leaky_relu = np.ElementwiseKernel(
        'float64 x',
        'float64 y',
        'y = (x * (x > 0)) + ((x <= 0) * x * 0.01)',
        'leakyurelu')
def dleaky_relu(x):
    return np.where(x > 0, 1, 0.01)


tanh = np.tanh
def dtanh(x):
    t = tanh(x)
    return 1-(t*t)

def no_activation(x):
    raise Exception("activation function not found or not implemented")

ACTIVATIONS ={
    "sigmoid": (sigmoid, dsigmoid),
    "relu": (relu, drelu),
    "leaky relu": (leaky_relu, dleaky_relu),
    "tanh": (tanh, dtanh),
}

## cost functions
def diff_squares(y, y_true):
    #print(f"diff squares of: {y.shape} - {y_true.shape}")
    return np.sum(np.square(y-y_true))
def ddiff_squares(y, y_true):
    return 2*(y-y_true)

def quadratic(y, y_true):
    return 0.5*diff_squares(y,y_true)
def dquadratic(y, y_true):
    return y-y_true

def cross_entropy(y, y_true):
    return -1*np.sum( y_true*np.log(y) + (1-y_true)*np.log(1-y) )
def dcross_entropy(y, y_true):
    return (y-y_true)/(y*(1-y))

def hellinger(y, y_true):
    return 0.70710678*np.sum( np.square(np.sqrt(y)-np.sqrt(y_true)) )
def dhellinger(y, y_true):
    rooty = np.sqrt(y)
    return (rooty - np.sqrt(y_true))/(1.41421356*rooty)

def no_cost(x):
    raise Exception("cost function not found or not implemented")
COSTS = {
    "diff_squares": (diff_squares, ddiff_squares),
    "quadratic": (quadratic, dquadratic),
    "cross-entropy": (cross_entropy, dcross_entropy),
    "hellinger": (hellinger, dhellinger),
}
## regularization functions

## synapse functions
##  probably won't touch these

## normalization functions


## reference functions
def classic_net_predict(weights, biases, input):
    for b, w in zip(biases, weights):
        # print(f"calculating: {w.shape} . {input.shape}")
        # print(f"weight: {w}\n")
        # print(f"input: {input}\n")
        input = sigmoid(np.dot(w, input)+b)
        # print(f"\t=> {input.shape}")
        # print(f"output: {input}")
    return input

def classic_net_backprop(weights, biases, input, exp_out):
    #TODO: turn this into a compiled NN class!
    #forward prop while recording
    activations = [input] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        zs.append(np.dot(w, activations[-1])+b)
        activations.append(sigmoid(zs[-1]))

    #error calc
    delta = ddiff_squares(activations[-1], exp_out)
    # print(f"\ndE = {delta.shape}")

    #gradient calc
    # print(f"first de[-1] = {delta.shape} * dsig")
    delta = delta * dsigmoid(zs[-1])
    # print(f"\t=> {delta.shape}")
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    nabla_b[-1] = delta
    # print(f"first weight = de {delta.shape} . {activations[-2].T.shape}")
    nabla_w[-1] = np.dot(delta, activations[-2].T)
    for l in range(2, len(weights)+1):
        nabla_b[-l] = np.dot(weights[-l+1].T, nabla_b[-l+1]) * dsigmoid(zs[-l])
        # print(f"db[{-l}] {nabla_b[-l].shape} = (weight.T {weights[-l+1].T.shape} . de {nabla_b[-l+1].shape}) * dsig")
        nabla_w[-l] = np.dot(nabla_b[-l], activations[-l-1].T)
        # print(f"dw[{-l}] {nabla_w[-l].shape} = db {nabla_b[-l].shape} . output[{2-l-1}] {activations[-l-1].T.shape})")
    # print(nabla_w[0])
    return (nabla_b, nabla_w)


def shuffle_in_unison(a, b):
    assert(len(a) == len(b))
    p = np.random.permutation(len(a))
    a[:] = a[p]
    b[:] = b[p]

def binarize(y, num_classes):
    targets = np.zeros((len(y),num_classes), np.float32)
    for i in range(targets.shape[0]):
        targets[i][y[i]] = 1
    return targets