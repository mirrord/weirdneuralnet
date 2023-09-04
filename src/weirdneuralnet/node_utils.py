# import numpy as np
import cupy as np

# #### activation functions
# from scipy.special import expit as sigmoid
sigmoid = np.ElementwiseKernel(
    "float64 x", "float64 y", "y = 1 / (1 + exp(-x))", "expit"
)


def dsigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


relu = np.ElementwiseKernel("float64 x", "float64 y", "y = x * (x > 0)", "relu")


def drelu(x):
    return np.greater(x, 0).astype("float64")


leaky_relu = np.ElementwiseKernel(
    "float64 x", "float64 y", "y = (x * (x > 0)) + ((x <= 0) * x * 0.01)", "leakyurelu"
)


def dleaky_relu(x):
    return np.where(x > 0, 1, 0.01)


tanh = np.tanh


def dtanh(x):
    t = tanh(x)
    return 1 - (t * t)


def softmax(x):
    s = np.max(x, axis=0)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=0)
    div = div[:, np.newaxis]  # ditto
    return e_x / div


def dsoftmax(x):
    sm = softmax(x)
    return sm * (np.eye(x.shape[0]) - sm.T)


def swish(x):
    return x * sigmoid(x)


def dswish(x):
    sigx = sigmoid(x)
    swishx = x * sigx
    return swishx + (sigx * (1 - swishx))


def scale_act(x):
    return x / x.shape[1]


def dscale(x):
    return 1 / x.shape[1]


def norm(x):
    return x / np.max(x, axis=1, keepdims=True)


def dnorm(x):
    return 1 / np.max(x, axis=1, keepdims=True)


def l2norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def dl2norm(x):
    return 1 / np.linalg.norm(x, axis=1, keepdims=True)


def no_activation(x):
    return x


# TODO: wrap these and log a warning when label is not found
ACTIVATIONS = {
    "sigmoid": (sigmoid, dsigmoid),
    "relu": (relu, drelu),
    "leaky relu": (leaky_relu, dleaky_relu),
    "tanh": (tanh, dtanh),
    "softmax": (softmax, dsoftmax),
    "swish": (swish, dswish),
    "scale": (scale_act, dscale),
    "norm": (norm, dnorm),
    "l2norm": (l2norm, dl2norm),
}

# #### cost functions
def diff_squares(y, y_true):
    # print(f"diff squares of: {y.shape} - {y_true.shape}")
    return np.sum(np.square(y - y_true)).item()


def ddiff_squares(y, y_true):
    return 2 * (y - y_true)


def quadratic(y, y_true):
    return 0.5 * diff_squares(y, y_true)


def dquadratic(y, y_true):
    return y - y_true


def cross_entropy(y, y_true):
    return -1 * np.sum(y_true * np.log(y) + (1 - y_true) * np.log(1 - y)).item()


def dcross_entropy(y, y_true):
    return (y - y_true) / (y * (1 - y))


def hellinger(y, y_true):
    return 0.70710678 * np.sum(np.square(np.sqrt(y) - np.sqrt(y_true))).item()


def dhellinger(y, y_true):
    rooty = np.sqrt(y)
    return (rooty - np.sqrt(y_true)) / (1.41421356 * rooty)


def no_cost(x):
    raise Exception("cost function not found or not implemented")


COSTS = {
    "diff_squares": (diff_squares, ddiff_squares),
    "quadratic": (quadratic, dquadratic),
    "cross-entropy": (cross_entropy, dcross_entropy),
    "hellinger": (hellinger, dhellinger),
}
# #### regularization functions
def ridge_reg(lamb):
    return lambda w: lamb * np.sum(np.abs(w)).item()


def lasso_reg(lamb):
    # generally considered to be more accurate and slower
    return lambda w: lamb * np.sum(np.square(w)).item()


def elastic_reg(alpha):
    L1 = ridge_reg(1 - alpha)
    L2 = lasso_reg(alpha)
    return lambda w: L1(w) + L2(w)


def noreg(_):
    return lambda w: 0


# these require hyperparameters!
REGULARIZATIONS = {
    "ridge": ridge_reg,
    "lasso": lasso_reg,
    "elastic": elastic_reg,
    "none": noreg,
}


# #### synapse functions
# #  probably won't touch these

# # normalization & standardization functions
def minmax_normalize(x):
    min = np.min(x).item()
    return (x - min) / (np.max(x).item() - min)


def minmaxneg_normalize(x):
    min = np.min(x).item()
    return 2 * (x - min) / (np.max(x).item() - min) - 1


def meanstd_standardize(x):
    return (x - np.mean(x).item()) / np.std(x).item()


def nonorm(x):
    return x


NORMALIZATIONS = {
    "minmax": minmax_normalize,
    "minmax-1": minmaxneg_normalize,
    "standard": meanstd_standardize,
    "none": nonorm,
}


# #### reference functions
def classic_net_predict(weights, biases, input):
    for b, w in zip(biases, weights):
        input = sigmoid(np.dot(input, w) + b)
    return input


def classic_net_backprop(weights, biases, input, exp_out):
    activations = [input]  # list to store all the activations, layer by layer
    zs = []  # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        zs.append(np.dot(activations[-1], w) + b)
        activations.append(sigmoid(zs[-1]))

    # error calc
    delta = ddiff_squares(activations[-1], exp_out)

    # gradient calc
    delta = delta * dsigmoid(zs[-1])
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(activations[-2].T, delta)
    for layer_idx in range(2, len(weights) + 1):
        nabla_b[-layer_idx] = np.dot(
            nabla_b[-layer_idx + 1], weights[-layer_idx + 1].T
        ) * dsigmoid(zs[-layer_idx])
        nabla_w[-layer_idx] = np.dot(activations[-layer_idx - 1].T, nabla_b[-layer_idx])
    return (nabla_b, nabla_w)


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    a[:] = a[p]
    b[:] = b[p]


def binarize(y, num_classes):
    y = y.astype(int)
    targets = np.zeros((len(y), num_classes), np.float32)
    for i in range(targets.shape[0]):
        targets[i][y[i]] = 1
    return targets


def debinarize(y):
    return y.argmax(axis=1)
