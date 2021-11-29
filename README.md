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
    for i in range(epochs):
        print(f"epoch: {i}...")
        model.train(X_train, Y_train)
        if i%10 == 0:
            epoch_list.append(i)
            costs.append(model.cost(model.get_last_output(), Y_train))
    # graph the cost during training
    plt.plot(epoch_list, costs)
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
