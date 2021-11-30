# weirdneuralnet
WeirdNeuralNet is a neural network library for exploring new topologies and training paradigms.

Create your own Weird Network:
```
from network import WeirdNetwork

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
requests
numpy
matplotlib
cupy

then just copy down this code.
There's no formal package yet. Maybe I'll get around to that at some point.
