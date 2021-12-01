

import cupy as cp
from time import time


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

funcs = drelu_funcs

def timeit(f):
    x = cp.random.randn(5000,5000)
    t = time()
    r = f(x)
    return time()-t

for name, func in funcs.items():
    t = 0
    for i in range(100):
        t+=timeit(func)
    print(f"{name}: {t/100} ms")