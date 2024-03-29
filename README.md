# WeirdNeuralNet

WeirdNeuralNet is a neural network library for exploring new topologies and training paradigms. It runs on any CUDA-enabled GPU.

## CLI
train a new Weird Network from scratch on the command line and save the result:
```
C:/users/JohnCleese> python -m weirdneuralnet --config network.cfg --train --epochs 100 --graph --save my_model.wnn
```
![example training cost plot](blob/Figure_1.jpg?raw=true)

#### TODO: add examples of loading & fine-tuning

## library interface
Create your own Weird Network in your project:
```
from weirdneuralnet import WeirdNetwork

node_params =[
        {
            'x':28*28,
            'y':128,
            'activation': 'sigmoid',
            'input':True
        },
        {
            'x':128,
            'y':10,
            'activation': 'sigmoid',
            'output':True
        }
    ]
edges = [
    (0,1),
]
model = WeirdNetwork(node_params, edges)
```

then train it:

```
cost_history = model.train(X_train, Y_train, epochs)
#plot cost history
plt.plot(list(range(epochs)), cost_history)
plt.show()
```

## install

requirements:
 * CUDA (see https://developer.nvidia.com/cuda-downloads)
 * Visual C++ Build Tools (version 14.2): https://visualstudio.microsoft.com/downloads/
    * direct link (will expire eventually): https://aka.ms/vs/17/release/vs_BuildTools.exe
 * CuPy (see https://pypi.org/project/cupy/ for pre-built binaries)

pull down this code, cd into the directory and run:
```pip install .```

This code has been tested using: python 3.7.8, CUDA 11.5, Windows 10