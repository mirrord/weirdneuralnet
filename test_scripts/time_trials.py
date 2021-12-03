

import cupy as cp
from time import time
from scipy.special import softmax


def kernelize(s, label):
    return cp.ElementwiseKernel(
        'float64 x',
        'float64 y',
        s,
        label)

# relu_funcs = {
#     "max": lambda x: cp.maximum(x, 0),
#     "in-place max": lambda x: cp.maximum(x, 0, x),
#     "mul": kernelize("y = x * (x > 0)", "mult"),
#     "abs": lambda x: (abs(x) + x) / 2
# }

def sep_drelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

drelu_funcs = {
    "separate": sep_drelu,
    "greater": lambda x: cp.greater(x, 0).astype('float64'),
    "where": lambda x: cp.where(x <= 0, 0, 1),
    "mult": lambda x: (x > 0) * 1.0
}

leaky_funcs = {
    "elementwise": kernelize('y = (x * (x > 0)) + ((x <= 0) * x * 0.01)', 'leakyurelu'),
    "maximum": lambda x: cp.maximum(0.1*x,x),
    "where": lambda x: cp.where(x > 0, x, x * 0.01)
}

def oneslike(x):
    dx = cp.ones_like(x)
    dx[x < 0] = 0.1
    return dx
dleaky = {
    "oneslike": oneslike,
    "where": lambda x: cp.where(x > 0, 1, 0.01)
}

def softmax_so(x):
    s = cp.max(x, axis=1)
    s = s[:, cp.newaxis] # necessary step to do broadcasting
    e_x = cp.exp(x - s)
    div = cp.sum(e_x, axis=1)
    div = div[:, cp.newaxis] # ditto
    return e_x / div

softmax_funcs = {
    "scipy": lambda x: softmax(x.get()),
    "stack overflow": softmax_so
}

def dsoftmax_eye(x):
    sm = softmax_so(x)
    return sm * (cp.eye(x.shape[0]) - sm.T)

dsoftmax_funcs = {
    "eye": dsoftmax_eye
}

funcs = dsoftmax_funcs

def timeit(f):
    x = cp.random.randn(5000,5000)
    t = time()
    r = f(x)
    return time()-t

def time_once():
    for name, func in funcs.items():
        print(f"{name}: {timeit(func)}")

def time_looped():
    times = {k:0 for k in funcs.keys()}
    for i in range(100):
        for name, func in funcs.items():
            times[name]+=timeit(func)
    for k, t in times.items():
        print(f"{k}: {t/100} ms")

def time_separate_looped():
    for name, func in funcs.items():
        t = 0
        for i in range(100):
            t+=timeit(func)
        print(f"{name}: {t/100} ms")

print("once:")
time_once()
print("\nlooped:")
time_looped()
